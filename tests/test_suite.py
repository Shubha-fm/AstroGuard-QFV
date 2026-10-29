"""Unit tests, synthetic fixtures not evidence."""
import tempfile
from pathlib import Path
import numpy as np
from suite.models import TwoLayer24,AffineGated24,Classifier
from suite.preprocess import encode_prefix,load_curve,quality_components,prepare,CurveStore
from suite.analyze import metric,quality
from tests.make_synthetic_fixture import build

def test_24_weights():
    for c in (TwoLayer24,AffineGated24):
        import torch
        m=c();x=torch.randn(3,4,requires_grad=True);out=m(x);out.sum().backward()
        assert out.shape==(3,4);assert sum(t.numel() for t in m.parameters())==24
        assert all(p.grad is not None for p in m.parameters())

def test_prepare_valid_splits_and_gate():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);build(root/'raw',18)
        r=prepare(root/'raw/objects.csv',root/'prepared');ds=CurveStore(root/'prepared')
        assert r['n_valid']==54
        train=set(ds.df.loc[ds.df.split=='train','object_id']);test=set(ds.df.loc[ds.df.split=='test','object_id'])
        assert not train.intersection(test)
        x,l,y=ds.batch(ds.indices('test'),prefix=5)
        assert x.shape[1:]==(128,6) and np.all(l==5)
        q=quality(ds.df.iloc[:5],(.5,.35,.15))
        assert np.all((q>=0)&(q<=1))

def test_model_shapes():
    import torch
    x=torch.randn(3,128,6);l=torch.tensor([20,30,128])
    for name in ('bigru','cnn','transformer','classical24a','classical24b'):
        m=Classifier(name,hidden=32,layers=1,heads=4)
        p=m(x,l)
        assert p.shape==(3,3)

def test_metrics_are_correct():
    y=np.array([0,1,2]);p=np.eye(3)
    assert metric(y,p)['macro_f1']==1.0
