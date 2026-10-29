"""Controlled corruptions of real test examples; NOT natural distribution shift.

Tests noise/dropout and band removal on retained physically measured curves.
"""
import time
from pathlib import Path
import numpy as np,pandas as pd
from .preprocess import CurveStore
from .train import softmax
from .common import CLASSES,seed_all,write_json

def stress(folder,prepared,out,seed=42,batch=32):
    import torch
    from .models import Classifier
    from sklearn.metrics import f1_score
    from .train import temperature
    folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ds=CurveStore(prepared);ix=ds.indices('test');x,ln,y=ds.batch(ix)
    results=[]
    paths=sorted(folder.rglob('checkpoint.pt'))
    if not paths:raise FileNotFoundError('No real PyTorch checkpoints found. Train the models first.')
    for path in paths:
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        model=Classifier(checkpoint['model'],checkpoint['hidden'],checkpoint['layers'],checkpoint['heads'])
        model.load_state_dict(checkpoint['model_state']);model.eval()
        for condition in ['clean','jitter_0.03','jitter_0.10','dropout_10pct','dropout_30pct','mask_g','mask_r']:
            rng=np.random.default_rng(seed);xc=x.copy();valid=np.arange(128)[None,:]<ln[:,None]
            if condition.startswith('jitter_'):
                sigma=float(condition.split('_')[1]);noise=rng.normal(0,sigma,xc[:,:,:3].shape);xc[:,:,:3]+=noise*valid[:,:,None]
            if condition.startswith('dropout_'):
                pct=float(condition.split('_')[1].replace('pct',''))/100.
                mask=(rng.random(valid.shape)<pct)&valid;xc[mask]=0.
            if condition.startswith('mask_'):
                band=3 if condition=='mask_g' else 4;mask=(xc[:,:,band]>0)&valid;xc[mask]=0.
            probs=[];start=time.perf_counter()
            with torch.no_grad():
                for st in range(0,len(ix),batch):
                    a=torch.from_numpy(xc[st:st+batch]);l=torch.from_numpy(ln[st:st+batch]);logits=model(a,l).numpy();probs.append(softmax(logits/checkpoint['temperature']))
            p=np.vstack(probs);pred=p.argmax(1)
            results.append({'model':checkpoint['model'],'seed':checkpoint['seed'],'condition':condition,'n':len(ix),'accuracy':float((pred==y).mean()),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'evaluation_seconds':round(time.perf_counter()-start,3)})
            print(checkpoint['model'],condition,results[-1]['macro_f1'],flush=True)
    pd.DataFrame(results).to_csv(out/'controlled_corruptions.csv',index=False)
    write_json(out/'stress_provenance.json',{'evidence':'Stress on held-out, real light curves, retrospectively perturbed; no evidence of true OOD transfer or live-ZTF deployment','n_test':len(ix)})
    return len(results)
