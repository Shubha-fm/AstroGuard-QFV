"""Scientific outputs from REAL held-out per-object model predictions only.

Repeated random seeds are not treated as independent test objects. Paired bootstrap
is conditional on one trained model/seed and a shared fixed object test set.
"""
import json,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import binomtest,ttest_rel
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,confusion_matrix
from .common import CLASSES,config,write_json

PROBCOLS=['p_'+c for c in CLASSES]

def metric(y,p):
    y=np.asarray(y);p=np.asarray(p,dtype=float);z=p.argmax(1)
    out={'n':len(y),'accuracy':float(accuracy_score(y,z)),'balanced_accuracy':float(balanced_accuracy_score(y,z)),'macro_f1':float(f1_score(y,z,labels=np.arange(3),average='macro',zero_division=0)), 'macro_precision':float(precision_score(y,z,labels=np.arange(3),average='macro',zero_division=0)), 'macro_recall':float(recall_score(y,z,labels=np.arange(3),average='macro',zero_division=0)), 'brier':float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))), 'log_loss':float(-np.mean(np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))))}
    conf=p.max(1);correct=(z==y)
    bins=np.minimum((conf*15).astype(int),14);ece=0
    for j in range(15):
        mask=bins==j
        if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
    out['ece_15']=float(ece)
    try:out['macro_auroc_ovr']=float(roc_auc_score(y,np.eye(3)[y]*0+p,multi_class='ovr',average='macro',labels=[0,1,2]))
    except ValueError:out['macro_auroc_ovr']=None
    return out

def safe_metrics(y,p):
    if len(y)<2:return None
    return metric(y,p)

def bootstrap_difference(y,a,b,iterations,seed):
    rng=np.random.default_rng(seed);n=len(y);samples=np.zeros(iterations,dtype=float)
    for i in range(iterations):
        ix=rng.integers(0,n,n)
        samples[i]=f1_score(y[ix],a[ix],labels=np.arange(3),average='macro',zero_division=0)-f1_score(y[ix],b[ix],labels=np.arange(3),average='macro',zero_division=0)
    return [float(x) for x in np.percentile(samples,[2.5,50,97.5])]

def quality(df,weight):
    obs=np.minimum(df.n_valid.to_numpy(dtype=float)/128.,1.)
    phot=np.exp(-df.median_mag_err.to_numpy(dtype=float))
    bands=((df.n_g.to_numpy(dtype=float)>0)&(df.n_r.to_numpy(dtype=float)>0)).astype(float)
    return np.clip(weight[0]*obs+weight[1]*phot+weight[2]*bands,0.,1.)

def choose_thresholds(val,base):
    # Validation-only empirical selection: maximal coverage under <=5% selective risk.
    y=np.array([CLASSES.index(c) for c in val.broad_class]);pred=np.array([CLASSES.index(c) for c in val.prediction]);correct=y==pred
    conf=val.confidence.to_numpy(dtype=float);quality0=quality(val,base['weights'])
    candidates=[]
    for c in (0.50,0.60,0.70,0.80,0.90,0.95):
      for q in (0.2,0.3,0.4,0.5,0.6):
        mask=(conf>=c)&(quality0>=q)&val.source_verified.astype(str).str.lower().isin(('true','1','yes')).to_numpy()
        if mask.any() and (1-correct[mask].mean())<=.05:
            candidates.append((float(mask.mean()),c,q,float(1-correct[mask].mean())))
    if not candidates:return {'selected':False,'confidence':base['confidence'],'quality':base['quality'],'note':'No configuration meets validation selective risk <=5% with at least one release'}
    best=sorted(candidates,key=lambda z:(z[0],-z[1],-z[2]),reverse=True)[0]
    return {'selected':True,'confidence':best[1],'quality':best[2],'validation_coverage':best[0],'validation_risk':best[3]}

def run_analysis(prediction_root,out,bootstrap=1000):
    prediction_root=Path(prediction_root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    paths=sorted(prediction_root.rglob('predictions.csv'))
    if not paths:raise FileNotFoundError('No REAL model predictions.csv found; execute train stage first')
    frames=[]
    for path in paths:
        x=pd.read_csv(path,dtype={'object_id':'string'})
        for c in ['object_id','broad_class','split','model','seed','prefix',*PROBCOLS,'confidence','n_valid','median_mag_err','n_g','n_r','source_verified','label_source']:
            if c not in x:raise ValueError(f'{path} missing {c}')
        x['seed']=pd.to_numeric(x.seed).astype(int);x['prefix']=pd.to_numeric(x.prefix).astype(int)
        frames.append(x)
    frame=pd.concat(frames,ignore_index=True)
    if frame.duplicated(['object_id','split','model','seed','prefix']).any():raise ValueError('Duplicate prediction rows within model/seed/prefix/split')
    if not set(frame.broad_class).issubset(set(CLASSES)):raise ValueError('Unknown labels')
    if frame[PROBCOLS].isna().any().any() or not np.allclose(frame[PROBCOLS].sum(axis=1),1,atol=1e-4):raise ValueError('Bad probabilities')
    original=frame[['object_id','split','broad_class']].drop_duplicates()
    if original.duplicated(['object_id','split']).any():raise ValueError('Conflicting true labels across runs')
    if original.groupby('object_id').split.nunique().max()>1:raise ValueError('Leakage: same object appears in different splits')
    weights=config()['quality_grid'];gate=config()['gate']
    base={'confidence':gate['confidence'],'quality':gate['quality'],'weights':config()['quality_weights']}
    rows=[];sens=[];subgroups=[];confusions=[];reliability=[];risks=[];selection=[]
    full=frame.groupby(['model','seed','prefix'],sort=True)
    for (model,seed,prefix),block in full:
        test=block[block.split=='test'].sort_values('object_id').copy()
        val=block[block.split=='val'].sort_values('object_id').copy()
        if test.empty or val.empty:continue
        y=np.array([CLASSES.index(c) for c in test.broad_class]);p=test[PROBCOLS].to_numpy(dtype=float)
        m=metric(y,p);rows.append({'model':model,'seed':seed,'prefix':prefix,**m,'mean_inference_ms_per_object':test.mean_inference_ms_per_object.mean()})
        cm=confusion_matrix(y,p.argmax(1),labels=[0,1,2]);confusions.extend({'model':model,'seed':seed,'prefix':prefix,'true_class':CLASSES[i],'pred_class':CLASSES[j],'count':int(cm[i,j])} for i in range(3) for j in range(3))
        if prefix==128:
            conf=p.max(1);pred=p.argmax(1);z=(pred==y)
            for b in range(15):
                mask=(conf>=b/15)&(conf<((b+1)/15) if b!=14 else conf<=1)
                if mask.any():reliability.append({'model':model,'seed':seed,'bin':b,'count':int(mask.sum()),'confidence':float(conf[mask].mean()),'accuracy':float(z[mask].mean())})
            rank=np.argsort(-conf)
            for coverage in np.linspace(.05,1,20):
                take=rank[:max(1,int(round(len(rank)*coverage)))];risks.append({'model':model,'seed':seed,'coverage':float(len(take)/len(rank)),'selective_risk':float(1-z[take].mean())})
        # Empirical observational quality sensitivity with fixed TEST objects and fixed model predictions.
        for w in weights:
            s=quality(test,w)
            conf=test.confidence.to_numpy(dtype=float)
            provenance=test.source_verified.astype(str).str.lower().isin(('true','1','yes')).to_numpy()
            accepted=(s>=gate['quality'])&(conf>=gate['confidence'])&provenance
            if accepted.any():selective_risk=float(np.mean(p.argmax(1)[accepted]!=y[accepted]))
            else:selective_risk=None
            sens.append({'model':model,'seed':seed,'prefix':prefix,'weights':','.join(map(str,w)), 'accepted':int(accepted.sum()),'coverage':float(accepted.mean()),'selective_risk':selective_risk,'note':'Confidence+quality+source only; epistemic gate requires multi-seed ensemble'})
        if prefix==128:
            selected=choose_thresholds(val,base)
            confidence_cut=selected['confidence'];quality_cut=selected['quality'];provenance=test.source_verified.astype(str).str.lower().isin(('true','1','yes')).to_numpy()
            accepted=provenance&(test.confidence.to_numpy()>=confidence_cut)&(quality(test,base['weights'])>=quality_cut)
            selection.append({'model':model,'seed':seed,**selected,'test_coverage':float(accepted.mean()),'test_risk':float((p.argmax(1)[accepted]!=y[accepted]).mean()) if accepted.any() else None,'test_releases':int(accepted.sum())})
            for name,mask in [('single_band',(test.n_g<=0)|(test.n_r<=0)),('both_bands',(test.n_g>0)&(test.n_r>0)),('minimum_20_to_39',test.n_valid.between(20,39)),('long_80_plus',test.n_valid>=80),('independent_labels',~test.label_source.astype(str).str.lower().str.contains('broker|alerce|pseudo|machine')),('broker_labels',test.label_source.astype(str).str.lower().str.contains('broker|alerce|pseudo|machine'))]:
                ix=np.asarray(mask,dtype=bool);score=safe_metrics(y[ix],p[ix])
                subgroups.append({'model':model,'seed':seed,'group':name,'n':int(ix.sum()),'accuracy':score['accuracy'] if score else None,'macro_f1':score['macro_f1'] if score else None})
            for subclass,grp in test.groupby('original_class'):
                j=grp.index.to_numpy();ix=test.index.isin(j);score=safe_metrics(y[ix],p[ix]);subgroups.append({'model':model,'seed':seed,'group':'broker_subclass_'+str(subclass),'n':int(ix.sum()),'accuracy':score['accuracy'] if score else None,'macro_f1':score['macro_f1'] if score else None})
    if not rows:raise ValueError('No matched validation/test predictions')
    pd.DataFrame(rows).to_csv(out/'model_metrics.csv',index=False)
    pd.DataFrame(confusions).to_csv(out/'confusion_counts.csv',index=False)
    pd.DataFrame(reliability).to_csv(out/'reliability_bins.csv',index=False)
    pd.DataFrame(risks).to_csv(out/'risk_coverage.csv',index=False)
    pd.DataFrame(sens).to_csv(out/'quality_weight_sensitivity.csv',index=False)
    pd.DataFrame(subgroups).to_csv(out/'subgroup_metrics.csv',index=False)
    pd.DataFrame(selection).to_csv(out/'validation_selected_gate.csv',index=False)
    # Paired same-seed same-object comparisons at full sequence only.
    comparisons=[]
    f=frame[(frame.split=='test')&(frame.prefix==128)]
    baseline_names=('rf','bigru','cnn','transformer','classical24a','classical24b')
    for seed in sorted(f.seed.unique()):
        group=f[f.seed==seed]
        q=group[group.model=='quantum24'].sort_values('object_id').reset_index(drop=True)
        if q.empty:continue
        for name in baseline_names:
            b=group[group.model==name].sort_values('object_id').reset_index(drop=True)
            if b.empty:continue
            merged=q[['object_id','broad_class','prediction']].merge(b[['object_id','broad_class','prediction']],on=['object_id','broad_class'],suffixes=('_quantum','_classical'),validate='one_to_one')
            if len(merged)!=len(q) or len(merged)!=len(b):raise ValueError(f'Comparison not exactly paired, seed={seed} baseline={name}')
            y=np.array([CLASSES.index(c) for c in merged.broad_class]);qa=np.array([CLASSES.index(c) for c in merged.prediction_quantum]);cb=np.array([CLASSES.index(c) for c in merged.prediction_classical]);qc=qa==y;bc=cb==y
            b01=int(np.sum(qc&~bc));b10=int(np.sum(~qc&bc));discord=b01+b10
            pval=float(binomtest(min(b01,b10),discord,.5,alternative='two-sided').pvalue) if discord else 1.
            qf=f1_score(y,qa,labels=[0,1,2],average='macro');bf=f1_score(y,cb,labels=[0,1,2],average='macro')
            ci=bootstrap_difference(y,qa,cb,bootstrap,config()['reporting']['bootstrap_seed']+int(seed))
            comparisons.append({'seed':int(seed),'baseline':name,'n_paired':len(merged),'quantum_macro_f1':float(qf),'baseline_macro_f1':float(bf),'delta_macro_f1':float(qf-bf),'bootstrap95_low':ci[0],'bootstrap95_high':ci[2],'mcnemar_exact_p':pval,'quantum_only_correct':b01,'baseline_only_correct':b10})
    if comparisons:
        cmp=pd.DataFrame(comparisons)
        # Holm for family-wise comparison of multiple classical baselines PER seed.
        ps=cmp.groupby('seed',group_keys=False)
        def holm(group):
            ix=np.argsort(group.mcnemar_exact_p.to_numpy());p=group.mcnemar_exact_p.to_numpy()[ix];adjust=np.maximum.accumulate(np.minimum(1.,p*(len(p)-np.arange(len(p)))))
            arr=np.zeros(len(p));arr[ix]=adjust;group=group.copy();group['mcnemar_holm_p']=arr;return group
        ps.apply(holm).to_csv(out/'paired_quantum_vs_classical.csv',index=False)
        seed_summary=[]
        for name,chunk in cmp.groupby('baseline'):
            delta=chunk.delta_macro_f1.to_numpy();seed_summary.append({'baseline':name,'n_seeds':len(delta),'mean_seed_delta':float(delta.mean()),'std_seed_delta':float(delta.std(ddof=1)) if len(delta)>1 else None,'paired_seed_t_p':float(ttest_rel(chunk.quantum_macro_f1,chunk.baseline_macro_f1).pvalue) if len(delta)>=3 else None,'warning':'Seed-level n is small; object bootstrap does not represent retraining uncertainty'})
        pd.DataFrame(seed_summary).to_csv(out/'across_seed_comparison.csv',index=False)
    # Ensemble predictions and uncertainty: mean probability and variance across seeds, on EXACT same object IDs.
    ens=[];ensemble_rows=[]
    for (model,prefix,split),group in frame.groupby(['model','prefix','split']):
        nseeds=group.seed.nunique()
        if nseeds<2:continue
        if group.groupby('object_id').seed.nunique().nunique()!=1 or group.groupby('object_id').seed.nunique().iloc[0]!=nseeds:
            raise ValueError(f'Incomplete seed predictions {model},{prefix},{split}')
        for oid,b in group.groupby('object_id'):
            p=b[PROBCOLS].to_numpy(dtype=float)
            row=b.iloc[0].to_dict();row.update({'model':model+'_ensemble','seed':-1,'prefix':prefix,'split':split,'epistemic_uncertainty':float(p.var(axis=0).mean())})
            avg=p.mean(axis=0);row['prediction']=CLASSES[int(avg.argmax())];row['confidence']=float(avg.max());row['predictive_entropy']=float(-(avg*np.log(avg+1e-12)).sum());
            for j,key in enumerate(PROBCOLS):row[key]=float(avg[j])
            ensemble_rows.append(row)
        if split=='test':
            sub=[r for r in ensemble_rows if r['model']==model+'_ensemble' and r['prefix']==prefix and r['split']=='test']
            y=np.array([CLASSES.index(c) for c in (r['broad_class'] for r in sub)]);p=np.array([[r[c] for c in PROBCOLS] for r in sub]);ens.append({'model':model,'prefix':prefix,'seeds':nseeds,**metric(y,p)})
    if ensemble_rows:
        ensemble=pd.DataFrame(ensemble_rows);ensemble.to_csv(out/'ensemble_predictions.csv',index=False)
        pd.DataFrame(ens).to_csv(out/'ensemble_metrics.csv',index=False)
        gates=[]
        for (model,prefix),test in ensemble[ensemble.split=='test'].groupby(['model','prefix']):
            y=np.array([CLASSES.index(c) for c in test.broad_class]);p=test[PROBCOLS].to_numpy()
            approved=(test.confidence>=gate['confidence'])&(test.epistemic_uncertainty<=gate['epistemic_uncertainty'])&(quality(test,config()['quality_weights'])>=gate['quality'])&test.source_verified.astype(str).str.lower().isin(('true','1','yes'))
            gates.append({'model':model,'prefix':prefix,'n':len(test),'released':int(approved.sum()),'coverage':float(approved.mean()),'released_accuracy':float((p.argmax(1)[approved]==y[approved]).mean()) if approved.any() else None,'released_empirical_error_rate':float((p.argmax(1)[approved]!=y[approved]).mean()) if approved.any() else None})
        pd.DataFrame(gates).to_csv(out/'ensemble_4way_release_gate.csv',index=False)
    # Simple figures, source from the generated measurements only.
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        d=pd.DataFrame(rows);full=d[d.prefix==128]
        fig,ax=plt.subplots(figsize=(9,4));full.boxplot(column='macro_f1',by='model',ax=ax,rot=30);ax.set_title('Held-out macro-F1 across separately trained seeds');ax.set_ylabel('macro-F1');fig.suptitle('');fig.tight_layout();fig.savefig(out/'model_f1_by_seed.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4));avg=d.groupby(['prefix','model']).macro_f1.mean().unstack();avg.plot(ax=ax,marker='o');ax.set_xlabel('Observed detections (retrospective prefix)');ax.set_ylabel('Held-out macro-F1');ax.set_title('Early-prefix performance (20+ detection cohort)');fig.tight_layout();fig.savefig(out/'early_prefix_performance.png',dpi=180);plt.close(fig)
        from scipy.stats import t
        grouped=full.groupby('model').macro_f1.agg(['mean','std','count']).sort_index()
        grouped['ci_halfwidth']=grouped.apply(lambda r: float(t.ppf(.975,r['count']-1)*r['std']/np.sqrt(r['count'])) if r['count']>1 else np.nan,axis=1)
        grouped.to_csv(out/'seed_mean_f1_confidence_intervals.csv')
        if grouped['count'].ge(2).any():
            fig,ax=plt.subplots(figsize=(9,4));good=grouped.dropna(subset=['ci_halfwidth']);ax.errorbar(np.arange(len(good)),good['mean'],yerr=good['ci_halfwidth'],fmt='o',capsize=5);ax.set_xticks(np.arange(len(good)),good.index,rotation=30);ax.set_ylabel('Mean held-out macro-F1');ax.set_title('95% t-interval across independent model seeds');fig.tight_layout();fig.savefig(out/'seed_macro_f1_ci95.png',dpi=180);plt.close(fig)
    except Exception as e:write_json(out/'plot_error.json',{'plot_error':str(e)})
    write_json(out/'analysis_provenance.json',{'input_files':len(paths),'prediction_rows':len(frame),'models':sorted(frame.model.unique()),'seeds':sorted(int(x) for x in frame.seed.unique()),'bootstrap_samples':bootstrap,'note':'All statistics from real supplied predictions. The original labels may be broker-generated; test metrics against those labels are not independently verified astrophysical accuracy.'})
    return {'models':sorted(frame.model.unique()),'seeds':sorted(int(x) for x in frame.seed.unique()),'prediction_rows':len(frame),'paired_comparison_count':len(comparisons),'output':str(out)}
