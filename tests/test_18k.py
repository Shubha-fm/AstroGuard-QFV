"""Synthetic SOFTWARE tests only. Never cite test fixture as astronomy results."""
import json,sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from suite.collect import _connection
from suite.preprocess_18k import prepare
from run_18k_collect import collect_18k,counts

class FakeBroker:
    def query_detections(self,**kwargs):
        mjd=np.arange(30)+60000
        return pd.DataFrame({'mjd':mjd,'magpsf':np.linspace(17,17.3,30),
                             'sigmapsf':np.ones(30)*.1,'fid':[1,2]*15})


def generate(tmp_path, n_each=12):
    root=tmp_path/'input';root.mkdir();db=_connection(root/'collection_progress.sqlite3')
    with db:
        db.execute("INSERT INTO settings VALUES (?,?)",('identity',json.dumps({
            'classifier':'lc_classifier','version':'hierarchical_random_forest_1.0.0',
            'target_objects':240000,'page_size':500,'seed':42})))
        for broad in ('Periodic','Stochastic','Transient'):
            for i in range(n_each):
                oid=f'ZTF_TEST_{broad}_{i:03d}'
                db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                    (oid,broad,broad+'_test',i+1,f'{i:08d}_{broad}', '0.6'))
    db.close();return root


def test_18k_resume_and_freeze_without_overwriting_original(tmp_path):
    root=generate(tmp_path)
    one=collect_18k(root,tmp_path/'results',per_class=6,max_new=6,delay=0,client=FakeBroker())
    assert one['completion']=='PAUSED'
    old=(root/'collection_progress.sqlite3').read_bytes()
    two=collect_18k(root,tmp_path/'results',per_class=6,max_new=30,delay=0,client=FakeBroker())
    assert two['completion']=='READY_FOR_PREPARE'
    assert counts(_connection(root/'collection_progress.sqlite3'))=={'Periodic':6,'Stochastic':6,'Transient':6}
    final=root/'objects_18.csv'
    frame=pd.read_csv(final)
    assert len(frame)==18 and frame.object_id.is_unique
    assert sorted(frame.broad_class.value_counts())==[6,6,6]
    assert frame.independent_label_verified.eq(False).all()
    assert not (root/'objects.csv').exists()
    sha=two['selected_manifest']['sha256']
    again=collect_18k(root,tmp_path/'results',per_class=6,max_new=3,delay=0,client=FakeBroker())
    assert again['new_statuses']=={} and again['selected_manifest']['sha256']==sha


def test_65_15_20_stratified_split(tmp_path):
    root=generate(tmp_path)
    collect_18k(root,tmp_path/'results',per_class=12,max_new=100,delay=0,client=FakeBroker())
    report=prepare(root/'objects_36.csv',tmp_path/'prepared',seed=42)
    assert report['n_valid']==36
    splits=pd.read_csv(tmp_path/'prepared'/'objects_and_splits.csv')
    assert set(splits['split'])=={'train','val','test'}
    assert splits.object_id.is_unique
    assert len(splits[splits.split=='test'])>=7
    assert len(splits[splits.split=='train'])>=22
    assert all(splits.groupby('broad_class').split.nunique()==3)
    assert '65%' in report['split_protocol']
