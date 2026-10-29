"""Photometry preparation: preserve raw curves and pseudolabel provenance.

Object manifest columns:
object_id,broad_class,original_class,label_source,source_verified,photometry_file,
sampling_frame,query_rank,broker_probability,retrieved_at,first_mjd,external_label

CSV file columns: mjd (or jd), magpsf (or mag), sigmapsf (or magerr), fid (or filter).
"""
from pathlib import Path
import json
import os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from .common import CLASSES,ROOT,config,sha256,stable_key,write_json

NEEDED=['object_id','broad_class','original_class','label_source','photometry_file','sampling_frame']

def load_curve(p):
    df=pd.read_csv(p)
    rename={}
    for target,alternatives in {'mjd':['mjd','jd'],'magpsf':['magpsf','mag'],'sigmapsf':['sigmapsf','magerr'],'fid':['fid','filter']}.items():
        src=next((c for c in alternatives if c in df.columns),None)
        if src is None: raise ValueError(f'{p}: missing {target}')
        rename[src]=target
    d=df.rename(columns=rename)[['mjd','magpsf','sigmapsf','fid']].copy()
    if 'jd' in rename and rename['jd']=='mjd':d['mjd']=pd.to_numeric(d['mjd'],errors='coerce')-2400000.5
    for c in ('mjd','magpsf','sigmapsf'):d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['mjd','magpsf','sigmapsf'])
    d=d[(d.sigmapsf>0)&np.isfinite(d.sigmapsf)].sort_values('mjd')
    if d.mjd.nunique()<2: return None
    # Preserve the EARLIEST 128 observations, avoiding selection based on future events.
    # This matters especially for retrospective early-alert prefix comparisons.
    if len(d)>128:d=d.iloc[:128].copy()
    f=d.fid.astype(str).str.strip().str.lower()
    d['g']=f.isin(['1','g','ztf_g','ztfg']).astype('int8')
    d['r']=f.isin(['2','r','ztf_r','ztfr']).astype('int8')
    return d

def encode_prefix(raw,n):
    p=raw.iloc[:min(int(n),len(raw))]; t=p.mjd.to_numpy(dtype=np.float32)
    mag=p.magpsf.to_numpy(dtype=np.float32);err=p.sigmapsf.to_numpy(dtype=np.float32)
    span=max(float(t.max()-t.min()),1e-6)
    med=np.median(mag);scale=max(float(1.4826*np.median(np.abs(mag-med))),float(np.std(mag)),1e-6)
    data=np.column_stack(((t-t.min())/span,(mag-med)/scale,np.clip(err/max(float(np.median(err)),1e-6),0,10),p.g.to_numpy(),p.r.to_numpy(),np.ones(len(p))))
    out=np.zeros((128,6),dtype=np.float32);out[:len(data)]=data
    return out.astype('float32'),len(p)

def quality_components(raw,n):
    p=raw.iloc[:int(n)];return {'n_valid':len(p),'median_mag_err':float(p.sigmapsf.median()),'n_g':int(p.g.sum()),'n_r':int(p.r.sum()),'observation_coverage':min(len(p)/128,1), 'photometric_error_component':float(np.exp(-p.sigmapsf.median())), 'band_component':int((p.g.sum()>0) and (p.r.sum()>0))}


PREFIX_CACHE = (2,5,10,20,40,80,128)
QUALITY_COLUMNS = ('n_valid','median_mag_err','n_g','n_r','observation_coverage','photometric_error_component','band_component')

def curve_features(raw,prefix=128):
    r=raw.iloc[:int(prefix)]
    t=r.mjd.to_numpy();m=r.magpsf.to_numpy();e=r.sigmapsf.to_numpy();n=len(r)
    if n>=3:slope=np.polyfit((t-t.min())/max(np.ptp(t),1e-6),m,1)[0]
    else:slope=0.
    return [n,float(np.ptp(t)),float(np.mean(m)),float(np.std(m)),float(np.median(e)),
            float(np.std(e)),float(np.percentile(m,90)-np.percentile(m,10)),float(slope),
            float(r.g.mean()),float(r.r.mean()),
            float(np.median(m[-max(1,n//4):])-np.median(m[:max(1,n//4)]))]

def prepare(manifest,output,limit=None,seed=42):
    """Disk-backed sequence processing with crash-safe, restartable checkpoints.

    At 240k objects, float32 (N,128,6) requires ~703 MiB plus metadata.
    No giant list of numpy arrays or compressed NPZ is constructed in RAM.
    This version retains raw input photometry in its original location.
    """
    manifest=Path(manifest).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(manifest,dtype={'object_id':'string','original_class':'string'})
    missing=set(NEEDED)-set(df.columns)
    if missing:raise ValueError(f'Manifest missing {sorted(missing)}')
    if df.object_id.isna().any() or df.object_id.duplicated().any():raise ValueError('Duplicate or missing object_id; resolve label conflict before preparing')
    if not set(df.broad_class.astype(str)).issubset(set(CLASSES)):raise ValueError('Unknown broad classes')
    df['object_id']=df.object_id.astype(str)
    if df.label_source.astype(str).str.strip().eq('').any():raise ValueError('label_source must be populated')
    if limit is not None:
        df=df.assign(_order=df.object_id.map(lambda s:stable_key(s,seed))).sort_values('_order').head(int(limit)).drop(columns='_order')
    settings={'manifest_sha256':sha256(manifest),'seed':int(seed),'limit':limit,'input_count':len(df),
              'representation':'float32_disk_memmap_earliest_128_with_prefix_cache_v1','minimum_eligible_observations':20}
    settingsfile=output/'prepare_settings.json'
    if settingsfile.exists():
        previous=json.loads(settingsfile.read_text(encoding='utf8'))
        if previous!=settings:
            raise ValueError('Prepared cache from a DIFFERENT manifest, seed or limit. Use a different --prepared directory, never overwrite a prior split.')
        if (output/'dataset_report.json').exists() and (output/'objects_and_splits.csv').exists() and (output/'full_sequences.npy').exists():
            return json.loads((output/'dataset_report.json').read_text())
    else:write_json(settingsfile,settings)
    array_file=output/'full_sequences.npy';journal=output/'prepare_progress.jsonl'
    from numpy.lib.format import open_memmap
    if array_file.exists():
        mmap=open_memmap(array_file,mode='r+')
        if mmap.shape!=(len(df),128,6):raise ValueError('Prepared memmap shape disagrees with manifest length')
    else:
        mmap=open_memmap(array_file,mode='w+',dtype=np.float32,shape=(len(df),128,6))
    prefix_arrays={}
    for pref in PREFIX_CACHE[:-1]:
        path=output/f'prefix_{pref}.npy'
        prefix_arrays[pref]=open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=(len(df),pref,6))
    qualpath=output/'prefix_quality.npy'
    qualmmap=open_memmap(qualpath,mode='r+' if qualpath.exists() else 'w+',dtype=np.float32,shape=(len(df),len(PREFIX_CACHE),len(QUALITY_COLUMNS)))
    featpath=output/'features_by_prefix.npy'
    featmmap=open_memmap(featpath,mode='r+' if featpath.exists() else 'w+',dtype=np.float32,shape=(len(df),len(PREFIX_CACHE),11))
    completed={};rows=[];skipped=[]
    if journal.exists():
        for line in journal.read_text(encoding='utf8').splitlines():
            if not line.strip():continue
            v=json.loads(line);completed[v['object_id']]=v
            if v['status']=='ok':rows.append(v['row'])
            else:skipped.append({'object_id':v['object_id'],'reason':v['reason']})
    pending=[];flush_every=100
    def commit_pending():
        if not pending:return
        mmap.flush()
        for prefmap in prefix_arrays.values():prefmap.flush()
        qualmmap.flush();featmmap.flush()
        with journal.open('a',encoding='utf8') as f:
            for row in pending:f.write(json.dumps(row,default=str)+'\n')
            f.flush();os.fsync(f.fileno())
        pending.clear()
    try:
        for i,row in enumerate(df.to_dict('records')):
            oid=str(row['object_id'])
            if oid in completed:continue
            path=Path(str(row['photometry_file']))
            if not path.is_absolute():path=manifest.parent/path
            reason=None
            if not path.is_file():reason='photometry_file_missing'
            else:
                try:raw=load_curve(path)
                except Exception as e:reason='invalid_photometry '+str(e);raw=None
                if reason is None and (raw is None or len(raw)<20):reason='fewer_than_20_valid_detections'
            if reason is None:
                x,l=encode_prefix(raw,len(raw));q=quality_components(raw,len(raw))
                index=len(rows)
                mmap[index,:,:]=x
                for k,pref in enumerate(PREFIX_CACHE):
                    if pref!=128:
                        encoded,_=encode_prefix(raw,pref)
                        prefix_arrays[pref][index,:,:]=encoded[:pref]
                    components=quality_components(raw,min(pref,len(raw)))
                    qualmmap[index,k,:]=[components[v] for v in QUALITY_COLUMNS]
                    featmmap[index,k,:]=curve_features(raw,pref)
                record={**row,**q,'n_valid_full':len(raw),'first_mjd_observed':float(raw.mjd.min()),
                        'last_mjd_observed':float(raw.mjd.max()),'photometry_file':str(path),'photometry_sha256':sha256(path)}
                rows.append(record)
                pending.append({'object_id':oid,'status':'ok','row':record})
            else:
                skipped.append({'object_id':oid,'reason':reason})
                pending.append({'object_id':oid,'status':'skipped','reason':reason})
            completed[oid]=True
            if len(pending)>=flush_every:
                commit_pending()
                if len(completed)%5000<flush_every:print(f'Prepared {len(completed):,}/{len(df):,} input objects; valid {len(rows):,}',flush=True)
        commit_pending()
    finally:
        commit_pending();mmap.flush();del mmap
        for pfmap in prefix_arrays.values():pfmap.flush()
        prefix_arrays.clear();qualmmap.flush();featmmap.flush()
        del qualmmap,featmmap
    if not rows:raise ValueError('No valid objects: inspect manifest / photometry paths; no astronomy results produced')
    out=pd.DataFrame(rows)
    count=out.broad_class.value_counts();valid=[c for c in CLASSES if count.get(c,0)>=3]
    if len(valid)!=3:raise ValueError('Need >=3 valid objects in EACH broad class for stratified splitting')
    index=np.arange(len(out));strat=out.broad_class.to_numpy()
    train,temp=train_test_split(index,test_size=.30,stratify=strat,random_state=seed)
    val,test=train_test_split(temp,test_size=.50,stratify=strat[temp],random_state=seed)
    split=np.empty(len(out),dtype=object);split[train]='train';split[val]='val';split[test]='test'
    out['split']=split;out['label_is_independent']=~out.label_source.astype(str).str.lower().str.contains('broker|alerce|machine|pseudo')
    out['source_verified']=out.source_verified if 'source_verified' in out else False
    out=out.reset_index(drop=True)
    # Persist metadata via atomic replacements. mmap has allocation for all input
    # rows; CurveStore reads only the first len(metadata) valid rows.
    temp=output/'objects_and_splits.csv.tmp';out.to_csv(temp,index=False);temp.replace(output/'objects_and_splits.csv')
    pd.DataFrame(skipped,columns=['object_id','reason']).to_csv(output/'excluded_objects.csv',index=False)
    summary={'n_in_input':len(df),'n_valid':len(out),'excluded':len(skipped),'split_counts':out.split.value_counts().to_dict(),
             'broad_class_counts':count.to_dict(),'sampling_frames':out.sampling_frame.value_counts().to_dict(),
             'label_sources':out.label_source.value_counts().to_dict(),'manifest_sha256':sha256(manifest),
             'disk_sequences_file':'full_sequences.npy','disk_allocation_count':len(df),
             'representation':'memory-mapped float32 earliest 128 with 7 prefix, features and quality caches',
             'warning':'ALeRCE broker labels are pseudolabels. Class-conditional samples and minimum 20 detections are NOT representative of the natural ZTF stream.'}
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4));out.broad_class.value_counts().reindex(CLASSES,fill_value=0).plot.bar(ax=ax)
        ax.set_ylabel('Objects');ax.set_title('Eligible sampled objects by pseudo-label')
        fig.tight_layout();fig.savefig(output/'dataset_class_counts.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4));ax.hist(out.n_valid_full,bins=22)
        ax.set_xlabel('Valid detections retained (max 128)');ax.set_ylabel('Objects')
        fig.tight_layout();fig.savefig(output/'detection_count_distribution.png',dpi=180);plt.close(fig)
    except Exception as e:summary['plot_error']=str(e)
    write_json(output/'dataset_report.json',summary)
    return summary


class CurveStore:
    def __init__(self, folder):
        folder=Path(folder);self.df=pd.read_csv(folder/'objects_and_splits.csv',dtype={'object_id':'string'});self.root=folder
        if (folder/'full_sequences.npy').exists():
            self.x=np.load(folder/'full_sequences.npy',mmap_mode='r')[:len(self.df)]
            self.l=self.df.n_valid_full.to_numpy(dtype=int)
            self.y=np.array([CLASSES.index(v) for v in self.df.broad_class],dtype=int)
        else:
            a=np.load(folder/'full_sequences.npz');self.x=a['x'];self.l=a['lengths'];self.y=a['labels']
        if len(self.x)!=len(self.df):raise ValueError('Dataset array metadata mismatch')
        self.prefix_arrays={}
        for pref in PREFIX_CACHE[:-1]:
            path=folder/f'prefix_{pref}.npy'
            if path.exists():self.prefix_arrays[pref]=np.load(path,mmap_mode='r')[:len(self.df)]
        self.prefix_qualities=np.load(folder/'prefix_quality.npy',mmap_mode='r')[:len(self.df)] if (folder/'prefix_quality.npy').exists() else None
        self.feature_cache=np.load(folder/'features_by_prefix.npy',mmap_mode='r')[:len(self.df)] if (folder/'features_by_prefix.npy').exists() else None
    def indices(self,split):return np.flatnonzero(self.df.split.to_numpy()==split)
    def batch(self,indices,prefix=128):
        indices=np.asarray(indices,dtype=int)
        if prefix>=128:
            return np.asarray(self.x[indices],dtype=np.float32),np.minimum(self.l[indices],128),self.y[indices]
        if prefix in self.prefix_arrays:
            out=np.zeros((len(indices),128,6),dtype=np.float32)
            out[:,:prefix,:]=self.prefix_arrays[prefix][indices]
            return out,np.minimum(self.l[indices],prefix),self.y[indices]
        xs=[];ls=[]
        for i in indices:
            raw=load_curve(self.df.iloc[i].photometry_file)
            arr,n=encode_prefix(raw,prefix);xs.append(arr);ls.append(n)
        return np.stack(xs),np.array(ls),self.y[indices]
    def quality_at(self,indices,prefix=128):
        indices=np.asarray(indices,dtype=int)
        if self.prefix_qualities is not None and prefix in PREFIX_CACHE:
            vals=np.asarray(self.prefix_qualities[indices,PREFIX_CACHE.index(prefix),:])
            return {col:vals[:,i] for i,col in enumerate(QUALITY_COLUMNS)}
        sub=self.df.iloc[indices]
        qs=[quality_components(load_curve(f),min(prefix,int(n))) for f,n in zip(sub.photometry_file,sub.n_valid_full)]
        return {col:np.array([v[col] for v in qs]) for col in QUALITY_COLUMNS}
    def features(self,indices,prefix=128):
        indices=np.asarray(indices,dtype=int)
        if self.feature_cache is not None and prefix in PREFIX_CACHE:
            return np.asarray(self.feature_cache[indices,PREFIX_CACHE.index(prefix),:],dtype=np.float32)
        return np.asarray([curve_features(load_curve(self.df.iloc[i].photometry_file),prefix) for i in indices],dtype=np.float32)
