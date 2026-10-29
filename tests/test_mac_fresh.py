"""Tests a truly fresh candidate frame. Uses fake data, never astronomical evidence."""
import pandas as pd
import numpy as np
from mac_query_status import frame_status
from suite.collect import collect
from run_18k_collect import collect_18k

class FakeClient:
    def query_objects(self, **kw):
        subclass=kw['class_name'].replace(' ','_');page=kw['page'];size=kw['page_size']
        frame=pd.DataFrame({'oid':[f'ZTF_SYN_{subclass}_{i:04d}' for i in range(5)],
                            'probability':[0.9]*5})
        return frame.iloc[(page-1)*size:page*size].reset_index(drop=True)
    def query_detections(self, **kw):
        return pd.DataFrame({'mjd':np.arange(24)+60000,'magpsf':17+np.arange(24)*.01,
                             'sigmapsf':np.ones(24)*.12,'fid':[1,2]*12})


def test_mac_fresh_query_and_18k_collection(tmp_path):
    folder=tmp_path/'input';client=FakeClient()
    assert frame_status(folder,2)['state']=='NOT_INITIALIZED'
    a=collect(folder,pages=2,page_size=3,target_objects=18,max_queries=3,
              query_only=True,delay=0,client=client)
    assert a['status']=='PAUSED_QUERY'
    assert frame_status(folder,2)['state']=='QUERY_INCOMPLETE'
    b=collect(folder,pages=2,page_size=3,target_objects=18,max_queries=100,
              query_only=True,delay=0,client=client)
    assert b['status']=='PAGES_READY'
    assert frame_status(folder,2)['state']=='PAGES_READY'
    c=collect_18k(folder,tmp_path/'results',per_class=3,max_new=40,delay=0,client=client)
    assert c['completion']=='READY_FOR_PREPARE'
    assert c['selected_manifest']['n']==9
    assert sorted(c['selected_manifest']['by_class'].values())==[3,3,3]
    assert len(list((folder/'lightcurves').glob('*.csv')))>=9


def test_refresh_queue_after_enlarged_candidate_frame(tmp_path):
    from suite.collect import _connection
    from run_imbalanced_bulk import setup_queue
    db=_connection(tmp_path/'queue.sqlite3')
    with db:
        db.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?)",('ZTF_SYN_A','Transient','SNIa',1,'aaa','0.8'))
    assert setup_queue(db)==1
    with db:
        db.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?)",('ZTF_SYN_B','Periodic','LPV',1,'bbb','0.8'))
    assert setup_queue(db)==2
    with db:
        db.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?)",('ZTF_SYN_A','Stochastic','AGN',1,'aaa','0.7'))
    assert setup_queue(db)==2
    state=db.execute('SELECT status FROM bulk_queue WHERE oid=?',('ZTF_SYN_A',)).fetchone()[0]
    assert state=='CONFLICT'
    db.close()
