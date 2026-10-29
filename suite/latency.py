"""Measured local archive->features->model->gate latency, no live broker transport.

Includes CSV parsing + feature computation + single-object CPU inference, not
ALeRCE remote query time. Accurate operational SLA requires live deployment.
"""
import time,platform
from pathlib import Path
import numpy as np,pandas as pd
from .preprocess import CurveStore,load_curve,encode_prefix,quality_components
from .common import config,write_json

def gate(confidence,quality,source):
    p=config()['gate'];return bool(source and confidence>=p['confidence'] and quality>=p['quality'])

def bench(trained,prepared,out,count=100,warmup=10,seed=42):
    import torch,joblib
    from .models import Classifier
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ds=CurveStore(prepared);ix=ds.indices('test');rng=np.random.default_rng(seed)
    selected=rng.choice(ix,size=min(len(ix),count),replace=False).tolist()
    if not selected:raise ValueError('No test data')
    results=[];checkpoints=list(Path(trained).rglob('checkpoint.pt'))+list(Path(trained).rglob('rf.joblib'))
    if not checkpoints:raise FileNotFoundError('No model checkpoints found')
    for path in checkpoints:
        is_rf=path.name=='rf.joblib'
        if is_rf: model=joblib.load(path);variant='rf';run_seed=int(path.parent.name.replace('seed_',''))
        else:
            chk=torch.load(path,map_location='cpu',weights_only=True)
            model=Classifier(chk['model'],chk['hidden'],chk['layers'],chk['heads']).cpu();model.load_state_dict(chk['model_state']);model.eval()
            variant=chk['model'];run_seed=int(chk['seed'])
        # warmup with real heldout source rows, not the calibration/test measurements
        all_steps=[selected[i%len(selected)] for i in range(warmup)]+selected
        elapsed=[];n_ok=0
        for i,idx in enumerate(all_steps):
            row=ds.df.iloc[idx];start=time.perf_counter()
            raw=load_curve(row.photometry_file)
            if is_rf:
                t=raw.mjd.to_numpy();m=raw.magpsf.to_numpy();e=raw.sigmapsf.to_numpy();n=len(raw)
                slope=np.polyfit((t-t.min())/max(np.ptp(t),1e-6),m,1)[0] if n>=3 else 0.
                x=np.array([[n,float(np.ptp(t)),float(np.mean(m)),float(np.std(m)),float(np.median(e)),float(np.std(e)),float(np.percentile(m,90)-np.percentile(m,10)),float(slope),float(raw.g.mean()),float(raw.r.mean()),float(np.median(m[-max(1,n//4):])-np.median(m[:max(1,n//4)]))]],dtype='float32')
                p=model.predict_proba(x)[0];conf=float(np.max(p))
            else:
                x,n=encode_prefix(raw,len(raw))
                with torch.no_grad():
                    logits=model(torch.from_numpy(x[None,:,:]),torch.tensor([n]))
                    conf=float(torch.softmax(logits/chk['temperature'],dim=-1).max().item())
            q=quality_components(raw,len(raw));qq=.5*q['observation_coverage']+.35*q['photometric_error_component']+.15*q['band_component']
            verified=str(row.source_verified).lower() in ('true','1','yes')
            n_ok+=int(gate(conf,qq,verified)) if i>=warmup else 0
            if i>=warmup:elapsed.append((time.perf_counter()-start)*1000)
        mem=None
        try:
            import psutil,os
            mem=int(psutil.Process().memory_info().rss)
        except ImportError:pass
        results.append({'model':variant,'seed':run_seed,'n_real_objects':len(elapsed),'p50_ms':float(np.percentile(elapsed,50)),'p95_ms':float(np.percentile(elapsed,95)),'p99_ms':float(np.percentile(elapsed,99)),'mean_ms':float(np.mean(elapsed)),'max_ms':float(np.max(elapsed)),'released':n_ok,'process_rss_bytes':mem,'warning':'Local CSV read + preprocessing + model inference + gate, single-object CPU; excludes ALeRCE network, alert transport, cache retrieval, queue and telescope operations'})
        print('LATENCY',variant,run_seed,'p95 ms',round(results[-1]['p95_ms'],2),flush=True)
    pd.DataFrame(results).to_csv(out/'local_end_to_end_latency.csv',index=False)
    write_json(out/'latency_provenance.json',{'hardware':platform.platform(),'method':'Warmups exclude all recorded latencies; randomly select fixed heldout objects by seed; CSV read and preprocess included; no server/broker I/O.','n_objects_requested':count,'seed':seed})
    return len(results)
