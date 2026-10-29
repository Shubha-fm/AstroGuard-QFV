"""Real-data model training with fixed object-disjoint splits and per-seed records.

No mock or synthetic labels are used. Each model fits train, selects epoch and
optional temperature using val, evaluates test once, and saves per-object data.
"""
import json,time,os,platform
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .preprocess import CurveStore
from .common import CLASSES,config,seed_all,write_json,version_info

MODELS={'rf','bigru','cnn','transformer','classical24a','classical24b','quantum24'}

def softmax(a):
    a=np.asarray(a);v=np.exp(a-np.max(a,axis=-1,keepdims=True));return v/v.sum(axis=-1,keepdims=True)

def temperature(logits,y):
    from scipy.special import logsumexp
    def objective(logt):
        z=logits/np.exp(logt)
        return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y]))
    z=minimize_scalar(objective,bounds=(np.log(.3),np.log(5.)),method='bounded')
    return float(np.exp(z.x))

def torch_predict(model,ds,ix,prefix,batch,device):
    import torch
    model.eval();all_logits=[];seconds=0.0
    with torch.no_grad():
        for start in range(0,len(ix),batch):
            ids=ix[start:start+batch];x,l,_=ds.batch(ids,prefix)
            xt=torch.from_numpy(x).to(device);lt=torch.from_numpy(l).to(device)
            if device.type=='cuda':torch.cuda.synchronize()
            t=time.perf_counter()
            pred=model(xt,lt)
            if device.type=='cuda':torch.cuda.synchronize()
            seconds+=time.perf_counter()-t
            all_logits.append(pred.detach().cpu().numpy())
    return np.vstack(all_logits),seconds

def train_model(ds,model_name,seed,folder,epochs=40,batch=32,hidden=128,layers=4,heads=4,max_train=None,device='auto',prefixes=(20,128),objective='supervised'):
    from sklearn.metrics import accuracy_score
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);seed_all(seed)
    train_idx=ds.indices('train');val_idx=ds.indices('val');test_idx=ds.indices('test')
    if len(val_idx)==0 or len(test_idx)==0:raise ValueError('Validation/test empty')
    if max_train is not None and len(train_idx)>max_train:
        # Balanced pilot subsample from TRAIN only, not the validation or test set.
        rng=np.random.default_rng(seed);groups=[train_idx[ds.y[train_idx]==j] for j in range(3)]
        subs=[]
        for group in groups:
            n=min(len(group),max(1,max_train//3));subs.extend(rng.choice(group,n,replace=False).tolist())
        train_idx=np.array(sorted(subs))
    if objective not in ('supervised','multitask'):raise ValueError('Unsupported objective')
    report={'objective':('supervised_classification_only (RF has no auxiliary heads)' if model_name=='rf' else objective),'model':model_name,'seed':seed,'n_train':len(train_idx),'n_val':len(val_idx),'n_test':len(test_idx),'architecture':'new controlled supervised benchmark, NOT exact re-run of original multitask manuscript network'}
    report['data_manifest_hash']=ds.df[['object_id','split','broad_class']].sort_values('object_id').to_csv(index=False)
    import hashlib
    report['data_manifest_sha256']=hashlib.sha256(report.pop('data_manifest_hash').encode()).hexdigest()
    report['setup']={'epochs':epochs,'batch':batch,'hidden':hidden,'layers':layers,'heads':heads,'max_train':max_train,'prefixes':list(prefixes),'objective':objective}
    outpred=[]
    if model_name=='rf':
        t=time.perf_counter()
        rf=RandomForestClassifier(n_estimators=200,min_samples_leaf=2,n_jobs=-1,random_state=seed,class_weight='balanced_subsample')
        rf.fit(ds.features(train_idx),ds.y[train_idx]);report['train_seconds']=round(time.perf_counter()-t,4)
        report['parameter_count']='not comparable to neural weights'
        import joblib
        joblib.dump(rf,folder/'rf.joblib')
        for prefix in prefixes:
            for subset,ix in [('val',val_idx),('test',test_idx)]:
                t=time.perf_counter();proba=rf.predict_proba(ds.features(ix,prefix));secs=time.perf_counter()-t
                a=np.zeros((len(ix),3));a[:,rf.classes_]=proba
                outpred.append(prediction_table(ds,ix,a,model_name,seed,subset,prefix,secs,report['n_train']))
    else:
        import torch
        from torch import nn
        from .models import Classifier,parameter_count
        choice='cuda' if device=='auto' and torch.cuda.is_available() and model_name!='quantum24' else 'cpu' if device=='auto' else device
        if model_name=='quantum24' and choice!='cpu':raise ValueError('Quantum24 runs on CPU to avoid device mismatch')
        dev=torch.device(choice);model=Classifier(model_name,hidden,layers,heads).to(dev)
        report['parameter_count']=parameter_count(model)
        report['device']=str(dev)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
        best=-1.;best_state=None;patience=0;history=[]
        tic=time.perf_counter()
        for epoch in range(1,epochs+1):
            model.train();order=np.random.default_rng(seed+epoch).permutation(train_idx)
            losses=[]
            for start in range(0,len(order),batch):
                ids=order[start:start+batch];x,l,y=ds.batch(ids)
                xt=torch.from_numpy(x).to(dev);lt=torch.from_numpy(l).to(dev);yt=torch.from_numpy(y).long().to(dev)
                optimizer.zero_grad()
                if objective=='multitask':
                    # Same input corruption for all neural variants. Mask 20% of observed
                    # channels at real timesteps, dropout 10% of points, normalized jitter
                    # sigma .03. Reconstruct masked values from SAME window, not future.
                    valid=torch.arange(xt.shape[1],device=dev)[None,:]<lt[:,None]
                    def augment():
                        corrupted=xt.clone()
                        corrupted[:,:,:3]+=(torch.randn_like(corrupted[:,:,:3])*.03*valid[:,:,None])
                        drop=(torch.rand(valid.shape,device=dev)<.10)&valid
                        corrupted[drop]=0
                        mask=(torch.rand(corrupted.shape,device=dev)<.20)&valid[:,:,None]
                        return corrupted.masked_fill(mask,0),mask
                    x1,mask=augment();x2,_=augment()
                    logits,reconstruction,rep=model(x1,lt,auxiliary=True)
                    _,_,rep2=model(x2,lt,auxiliary=True)
                    ce=nn.functional.cross_entropy(logits,yt)
                    reconloss=nn.functional.mse_loss(reconstruction[mask],xt[mask]) if mask.any() else nn.functional.mse_loss(reconstruction[valid],xt[valid])
                    # Within-minibatch InfoNCE, augmented same-object positive pair.
                    z=torch.cat([rep,rep2],dim=0)
                    similarity=(z@z.T)/.20
                    similarity.fill_diagonal_(-1e9)
                    nr=rep.size(0)
                    positives=(torch.arange(2*nr,device=dev)+nr)%(2*nr)
                    contrast=nn.functional.cross_entropy(similarity,positives)
                    loss=.80*ce+.10*reconloss+.10*contrast
                else:
                    logits=model(xt,lt)
                    loss=nn.functional.cross_entropy(logits,yt)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
                losses.append(float(loss.detach().cpu()))
            val_logits,_=torch_predict(model,ds,val_idx,128,batch,dev)
            f1=float(f1_score(ds.y[val_idx],val_logits.argmax(1),average='macro',zero_division=0))
            history.append({'epoch':epoch,'loss':float(np.mean(losses)),'validation_macro_f1':f1})
            print(f'[{model_name} seed={seed}] epoch={epoch} loss={history[-1]["loss"]:.4f} val_f1={f1:.4f}',flush=True)
            if f1>best+1e-8:
                best=f1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};patience=0
            else:patience+=1
            if patience>=8:break
        report['train_seconds']=round(time.perf_counter()-tic,4);report['best_val_f1']=best;report['epochs_run']=len(history)
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        model.load_state_dict(best_state);model.eval()
        val_logits,_=torch_predict(model,ds,val_idx,128,batch,dev)
        tau=temperature(val_logits,ds.y[val_idx]);report['temperature_from_validation']=tau
        torch.save({'model_state':best_state,'model':model_name,'hidden':hidden,'layers':layers,'heads':heads,'seed':seed,'temperature':tau,'objective':objective},folder/'checkpoint.pt')
        for prefix in prefixes:
            for subset,ix in [('val',val_idx),('test',test_idx)]:
                logits,sec=torch_predict(model,ds,ix,prefix,batch,dev)
                outpred.append(prediction_table(ds,ix,softmax(logits/tau),model_name,seed,subset,prefix,sec,report['n_train']))
    allpred=pd.concat(outpred,ignore_index=True)
    allpred.to_csv(folder/'predictions.csv',index=False)
    write_json(folder/'train_summary.json',report)
    return report

def prediction_table(ds,ix,proba,model,seed,split,prefix,seconds,n_train):
    sub=ds.df.iloc[ix].copy().reset_index(drop=True)
    # Quality features for each prefix, to avoid using FUTURE observations in early alert gating.
    if prefix<128:
        components=ds.quality_at(ix,prefix)
        for k,v in components.items():sub[k]=v
    cols=['object_id','broad_class','original_class','label_source','label_is_independent','sampling_frame','split','n_valid','median_mag_err','n_g','n_r','source_verified','first_mjd_observed','last_mjd_observed','observation_coverage','photometric_error_component','band_component']
    sub=sub[[x for x in cols if x in sub.columns]].copy()
    proba=np.asarray(proba,dtype=float)
    if proba.shape!=(len(ix),3):raise ValueError('Probabilities must have shape [N,3]')
    if not np.all(np.isfinite(proba)) or not np.allclose(proba.sum(axis=1),1,atol=1e-4):raise ValueError('Invalid class probabilities')
    sub['model']=model;sub['seed']=seed;sub['prefix']=prefix;sub['trained_n']=n_train
    sub['prediction']=np.array(CLASSES)[proba.argmax(1)];sub['confidence']=proba.max(1)
    sub['predictive_entropy']=-(proba*np.log(proba+1e-12)).sum(1)
    for j,c in enumerate(CLASSES):sub[f'p_{c}']=proba[:,j]
    sub['batch_inference_seconds']=float(seconds)
    sub['mean_inference_ms_per_object']=1000*seconds/max(len(ix),1)
    return sub
