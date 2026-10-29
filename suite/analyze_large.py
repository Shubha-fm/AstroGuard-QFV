"""Memory-bounded reviewer statistics for up to 240k astronomical objects.

Reads one model-seed predictions CSV at a time (not 35 combined CSVs).
Bootstrap on the 27 joint outcome categories is EXACTLY equivalent in
sampling distribution to an ordinary paired nonparametric object bootstrap.
Full-sequence ensembles are produced; all prefix-by-model metrics remain.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel, t
from sklearn.metrics import confusion_matrix, f1_score
from .common import CLASSES, config, write_json
from .analyze import metric, quality, choose_thresholds, PROBCOLS


def bootstrap_27(y, pred_a, pred_b, n_boot=3000, seed=20261026):
    y=np.asarray(y,dtype=int); a=np.asarray(pred_a,dtype=int);b=np.asarray(pred_b,dtype=int)
    if len(y)==0:raise ValueError('Empty paired predictions')
    freq=np.bincount(y*9+a*3+b,minlength=27).astype(np.float64)
    rng=np.random.default_rng(seed)
    # Each bootstrap draw is a multinomial over the 27 possible observed
    # (true, quantum-prediction, classical-prediction) cells.
    counts=rng.multinomial(len(y),freq/len(y),size=n_boot).reshape((n_boot,3,3,3))
    q=counts.sum(axis=3);c=counts.sum(axis=2)
    def f1(conf):
        tp=np.diagonal(conf,axis1=1,axis2=2)
        denom=conf.sum(axis=1)+conf.sum(axis=2)
        return np.divide(2*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom>0).mean(axis=1)
    differences=f1(q)-f1(c)
    return np.quantile(differences,[.025,.5,.975]).tolist()


def _save(df,path):
    pd.DataFrame(df).to_csv(path,index=False)


def run_analysis_large(prediction_root,out,bootstrap=3000):
    prediction_root=Path(prediction_root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    paths=sorted(prediction_root.rglob('predictions.csv'))
    if not paths:raise FileNotFoundError('No executed model predictions.csv found; train first')
    cache=out/'_cache';cache.mkdir(exist_ok=True)
    weights=config()['quality_grid'];gate=config()['gate']
    base={'confidence':gate['confidence'],'quality':gate['quality'],'weights':config()['quality_weights']}
    rows=[];sens=[];subgroups=[];confusions=[];reliability=[];risks=[];selection=[]
    records={};input_rows=0;global_object_split={}
    for source in paths:
        frame=pd.read_csv(source,dtype={'object_id':'string'})
        required=['object_id','broad_class','split','model','seed','prefix',*PROBCOLS,'confidence',
                  'n_valid','median_mag_err','n_g','n_r','source_verified','label_source','prediction']
        missing=set(required)-set(frame.columns)
        if missing:raise ValueError(f'{source}: missing fields {sorted(missing)}')
        if frame.duplicated(['object_id','split','model','seed','prefix']).any():raise ValueError(f'{source}: duplicate prediction rows')
        if frame[PROBCOLS].isna().any().any() or not np.allclose(frame[PROBCOLS].sum(axis=1),1,atol=1e-4):raise ValueError(f'{source}: invalid probabilities')
        if not set(frame.broad_class).issubset(set(CLASSES)):raise ValueError(f'{source}: unknown truth labels')
        model_names=frame.model.unique();seeds=frame.seed.unique()
        if len(model_names)!=1 or len(seeds)!=1:raise ValueError(f'{source}: must describe exactly one model and seed')
        model=str(model_names[0]);seed=int(seeds[0]);key=(model,seed)
        if key in records:raise ValueError('Duplicate model-seed prediction files: '+str(key))
        input_rows+=len(frame)
        # Unique object + split + truth consistency is checked using an object
        # fingerprint per run, avoiding holding every large row in memory.
        unique=frame[['object_id','split','broad_class']].drop_duplicates()
        if unique.duplicated('object_id').any():raise ValueError(f'{source}: same object multiple splits / conflicting truth')
        digest=pd.util.hash_pandas_object(unique.sort_values('object_id').reset_index(drop=True),index=False).to_numpy(dtype=np.uint64)
        import hashlib
        fingerprint=hashlib.sha256(digest.tobytes()).hexdigest()
        if global_object_split and fingerprint!=next(iter(global_object_split.values())):
            raise ValueError(f'{source}: object IDs, splits, or truth labels differ across models; comparison would not be paired')
        global_object_split[key]=fingerprint
        for prefix,block in frame.groupby('prefix',sort=True):
            prefix=int(prefix)
            test=block[block.split=='test'].sort_values('object_id').copy()
            val=block[block.split=='val'].sort_values('object_id').copy()
            if len(test)==0 or len(val)==0:continue
            y=np.array([CLASSES.index(v) for v in test.broad_class]);p=test[PROBCOLS].to_numpy(dtype=float)
            m=metric(y,p)
            rows.append({'model':model,'seed':seed,'prefix':prefix,**m,
                         'mean_inference_ms_per_object':float(test.mean_inference_ms_per_object.mean()) if 'mean_inference_ms_per_object' in test else None})
            cm=confusion_matrix(y,p.argmax(1),labels=[0,1,2]);
            confusions.extend({'model':model,'seed':seed,'prefix':prefix,'true_class':CLASSES[i],'pred_class':CLASSES[j],'count':int(cm[i,j])} for i in range(3) for j in range(3))
            for weight in weights:
                s=quality(test,weight);approved=(s>=gate['quality'])&(test.confidence.to_numpy()>=gate['confidence'])&test.source_verified.astype(str).str.lower().isin(['true','1','yes']).to_numpy()
                sens.append({'model':model,'seed':seed,'prefix':prefix,'weights':','.join(map(str,weight)),
                             'accepted':int(approved.sum()),'coverage':float(approved.mean()),
                             'selective_risk':float((p.argmax(1)[approved]!=y[approved]).mean()) if approved.any() else None,
                             'note':'Fixed predictions; confidence+quality+source gate only; ensemble uncertainty reported separately'})
            if prefix!=128:continue
            # Persist only one small, full-sequence slice per model/seed for
            # cross-model comparisons and ensembles. Not loaded all at once.
            testpath=cache/f'{model}_seed{seed}_test128.pkl'
            valpath=cache/f'{model}_seed{seed}_val128.pkl'
            test.to_pickle(testpath);val.to_pickle(valpath)
            records[key]={'test':str(testpath),'val':str(valpath),'n_test':len(test),'n_val':len(val)}
            conf=p.max(1);pred=p.argmax(1);correct=pred==y
            for binidx in range(15):
                mask=(conf>=binidx/15)&(conf<((binidx+1)/15) if binidx!=14 else conf<=1)
                if mask.any():reliability.append({'model':model,'seed':seed,'bin':binidx,'count':int(mask.sum()),'confidence':float(conf[mask].mean()),'accuracy':float(correct[mask].mean())})
            rank=np.argsort(-conf)
            for cov in np.linspace(.05,1,20):
                chosen=rank[:max(1,int(round(len(rank)*cov)))];risks.append({'model':model,'seed':seed,'coverage':float(len(chosen)/len(rank)),'selective_risk':float(1-correct[chosen].mean())})
            chosen=choose_thresholds(val,base)
            passed=(test.confidence.to_numpy()>=chosen['confidence'])&(quality(test,base['weights'])>=chosen['quality'])&test.source_verified.astype(str).str.lower().isin(['true','1','yes']).to_numpy()
            selection.append({'model':model,'seed':seed,**chosen,'test_coverage':float(passed.mean()),
                             'test_risk':float((pred[passed]!=y[passed]).mean()) if passed.any() else None,'test_releases':int(passed.sum())})
            masks=[('single_band',(test.n_g<=0)|(test.n_r<=0)),('both_bands',(test.n_g>0)&(test.n_r>0)),
                   ('minimum_20_to_39',test.n_valid.between(20,39)),('long_80_plus',test.n_valid>=80),
                   ('independent_labels',~test.label_source.astype(str).str.lower().str.contains('broker|alerce|pseudo|machine')),
                   ('broker_labels',test.label_source.astype(str).str.lower().str.contains('broker|alerce|pseudo|machine'))]
            for name,mask in masks:
                ix=np.asarray(mask,dtype=bool)
                subgroups.append({'model':model,'seed':seed,'group':name,'n':int(ix.sum()),
                                  'accuracy':float((pred[ix]==y[ix]).mean()) if ix.any() else None,
                                  'macro_f1':float(f1_score(y[ix],pred[ix],labels=np.arange(3),average='macro',zero_division=0)) if ix.sum()>=3 else None})
            for subclass,grp in test.groupby('original_class'):
                ix=test.original_class.eq(subclass).to_numpy()
                subgroups.append({'model':model,'seed':seed,'group':'broker_subclass_'+str(subclass),
                                  'n':int(ix.sum()),'accuracy':float((pred[ix]==y[ix]).mean()) if ix.any() else None,
                                  'macro_f1':None})
        del frame
        print(f'Analyzed model={model}, seed={seed}; source={source}',flush=True)
    if not rows:raise ValueError('No matched validation/test predictions')
    for filename,data in [('model_metrics.csv',rows),('quality_weight_sensitivity.csv',sens),('confusion_counts.csv',confusions),
                          ('subgroup_metrics.csv',subgroups),('reliability_bins.csv',reliability),('risk_coverage.csv',risks),
                          ('validation_selected_gate.csv',selection)]:
        _save(data,out/filename)
    comparisons=[]
    for (model,seed),item in sorted(records.items()):
        if model!='quantum24':continue
        q=pd.read_pickle(item['test'])
        for baseline in ('rf','bigru','cnn','transformer','classical24a','classical24b'):
            if (baseline,seed) not in records:continue
            b=pd.read_pickle(records[(baseline,seed)]['test'])
            merged=q[['object_id','broad_class','prediction']].merge(b[['object_id','broad_class','prediction']],
                        on=['object_id','broad_class'],validate='one_to_one',suffixes=('_quantum','_classical'))
            if len(merged)!=len(q) or len(merged)!=len(b):raise ValueError('Paired comparison mismatched heldout objects')
            y=np.array([CLASSES.index(x) for x in merged.broad_class]);a=np.array([CLASSES.index(x) for x in merged.prediction_quantum]);c=np.array([CLASSES.index(x) for x in merged.prediction_classical])
            aq=a==y;bc=c==y;n01=int((aq&~bc).sum());n10=int((~aq&bc).sum());discord=n01+n10
            pv=float(binomtest(min(n01,n10),discord,.5).pvalue) if discord else 1.
            qa=f1_score(y,a,labels=[0,1,2],average='macro');bl=f1_score(y,c,labels=[0,1,2],average='macro')
            ci=bootstrap_27(y,a,c,bootstrap,config()['reporting']['bootstrap_seed']+seed)
            comparisons.append({'seed':seed,'baseline':baseline,'n_paired':len(y),'quantum_macro_f1':float(qa),
                                'baseline_macro_f1':float(bl),'delta_macro_f1':float(qa-bl),
                                'bootstrap95_low':ci[0],'bootstrap95_high':ci[2],
                                'mcnemar_exact_p':pv,'quantum_only_correct':n01,'baseline_only_correct':n10})
    if comparisons:
        cmp=pd.DataFrame(comparisons)
        adjusted=[]
        for seed,group in cmp.groupby('seed'):
            group=group.copy();idx=np.argsort(group.mcnemar_exact_p.to_numpy());pvals=group.mcnemar_exact_p.to_numpy()[idx]
            holm=np.maximum.accumulate(np.minimum(1.,pvals*(len(pvals)-np.arange(len(pvals)))))
            values=np.zeros(len(group));values[idx]=holm;group['mcnemar_holm_p']=values;adjusted.append(group)
        _save(pd.concat(adjusted,ignore_index=True),out/'paired_quantum_vs_classical.csv')
        summary=[]
        for baseline,group in cmp.groupby('baseline'):
            d=group.delta_macro_f1.to_numpy()
            summary.append({'baseline':baseline,'n_seeds':len(d),'mean_seed_delta':float(d.mean()),
                            'std_seed_delta':float(d.std(ddof=1)) if len(d)>1 else None,
                            'paired_seed_t_p':float(ttest_rel(group.quantum_macro_f1,group.baseline_macro_f1).pvalue) if len(d)>=3 else None,
                            'warning':'Paired object bootstrap conditions on trained weights; n_seeds is small'})
        _save(summary,out/'across_seed_comparison.csv')
    ensemble_stats=[];gate_stats=[]
    for model in sorted({m for m,s in records}):
        ss=sorted(sd for m,sd in records if m==model)
        if len(ss)<2:continue
        for split in ('test','val'):
            blocks=[pd.read_pickle(records[(model,sd)][split]) for sd in ss]
            base=blocks[0].sort_values('object_id').reset_index(drop=True)
            for other in blocks[1:]:
                other.sort_values('object_id',inplace=True)
                other.reset_index(drop=True,inplace=True)
                if not base[['object_id','broad_class']].equals(other[['object_id','broad_class']]):
                    raise ValueError(f'Incomplete paired ensemble objects: {model},{split}')
            prob=np.stack([p[PROBCOLS].to_numpy(dtype=np.float32) for p in blocks]);avg=prob.mean(0);epi=prob.var(0).mean(1)
            y=np.array([CLASSES.index(v) for v in base.broad_class])
            base['model']=model+'_ensemble';base['seed']=-1;base['confidence']=avg.max(1)
            base['epistemic_uncertainty']=epi;base['prediction']=np.asarray(CLASSES)[avg.argmax(1)]
            for j,c in enumerate(PROBCOLS):base[c]=avg[:,j]
            base.to_csv(out/f'ensemble_{model}_{split}_128.csv',index=False)
            if split!='test':continue
            ensemble_stats.append({'model':model,'prefix':128,'seeds':len(ss),**metric(y,avg)})
            accepted=(base.confidence.to_numpy()>=gate['confidence'])&(epi<=gate['epistemic_uncertainty'])&(
                quality(base,config()['quality_weights'])>=gate['quality'])&base.source_verified.astype(str).str.lower().isin(['true','1','yes']).to_numpy()
            gate_stats.append({'model':model,'prefix':128,'n':len(base),'released':int(accepted.sum()),
                               'coverage':float(accepted.mean()),'released_accuracy':float((avg.argmax(1)[accepted]==y[accepted]).mean()) if accepted.any() else None,
                               'released_empirical_error_rate':float((avg.argmax(1)[accepted]!=y[accepted]).mean()) if accepted.any() else None})
    if ensemble_stats:_save(ensemble_stats,out/'ensemble_metrics.csv')
    if gate_stats:_save(gate_stats,out/'ensemble_4way_release_gate.csv')
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        d=pd.DataFrame(rows);full=d[d.prefix==128]
        fig,ax=plt.subplots(figsize=(9,4));full.boxplot(column='macro_f1',by='model',ax=ax,rot=30)
        ax.set_title('Held-out macro-F1 across model seeds');ax.set_ylabel('macro-F1');fig.suptitle('');fig.tight_layout();fig.savefig(out/'model_f1_by_seed.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4));d.groupby(['prefix','model']).macro_f1.mean().unstack().plot(ax=ax,marker='o')
        ax.set_xlabel('Available detections (retrospective prefix)');ax.set_ylabel('Macro-F1');fig.tight_layout();fig.savefig(out/'early_prefix_performance.png',dpi=180);plt.close(fig)
        grouped=full.groupby('model').macro_f1.agg(['mean','std','count'])
        grouped['ci_halfwidth']=grouped.apply(lambda v:float(t.ppf(.975,v['count']-1)*v['std']/np.sqrt(v['count'])) if v['count']>1 else np.nan,axis=1)
        grouped.to_csv(out/'seed_mean_f1_confidence_intervals.csv')
    except Exception as e:write_json(out/'plot_error.json',{'exception':str(e)})
    write_json(out/'analysis_provenance.json',{'prediction_files':len(paths),'prediction_rows':input_rows,
               'model_seed_pairs':[{'model':m,'seed':s,'n_test':v['n_test']} for (m,s),v in records.items()],
               'bootstrap_samples':bootstrap,'bootstrap_implementation':'multinomial resampling of 27 joint class outcomes; equivalent distribution to object-level paired resampling',
               'memory_strategy':'CSV processed one at a time; cache only full-128 model slices on disk; ensembles full-128 only',
               'labels_warning':'ALeRCE pseudo-labels are not independently confirmed astrophysical truth.'})
    return {'model_seed_runs':len(records),'prediction_files':len(paths),'prediction_rows':input_rows,
            'paired_comparison_count':len(comparisons),'full_128_ensemble_models':len(ensemble_stats),'output':str(out)}
