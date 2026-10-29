import hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CLASSES=('Periodic','Stochastic','Transient')

def config():
    return json.loads((ROOT/'configs/protocol.json').read_text(encoding='utf8'))

def seed_all(seed):
    random.seed(int(seed)); np.random.seed(int(seed))
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
    except ImportError: pass

def write_json(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,indent=2,allow_nan=False,default=str)+'\n',encoding='utf8')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def stable_key(s,seed=42):
    return hashlib.sha256(f'{seed}|{s}'.encode()).hexdigest()

def version_info():
    import platform
    o={'python':sys.version,'platform':platform.platform(),'cpu_count':os.cpu_count()}
    for x in ('numpy','pandas','scipy','sklearn','torch','pennylane','alerce'):
        try:
            m=__import__(x);o[x]=getattr(m,'__version__','available')
        except Exception:o[x]='unavailable'
    return o
