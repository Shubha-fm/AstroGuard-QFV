"""Resumable, disk-backed ALeRCE/ZTF class-conditioned collection.

ALeRCE classes are *broker pseudo-labels*, not independently confirmed truth.
Pages are an ID-ordered, class-filtered frame; hash selection is deterministic
only WITHIN this finite queried frame and cannot estimate ZTF prevalence.

SQLite stores every queried candidate and every success/failure. Interrupted
runs resume without repeating completed pages or photometry downloads.
"""
import datetime
import json
import math
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from .common import config, stable_key, version_info, write_json

GROUPS = {
    'Transient': ['SN', 'SNIa', 'SNIbc', 'SNII'],
    'Periodic': ['Periodic-Other', 'RR Lyrae', 'Eclipsing Binary', 'LPV'],
    'Stochastic': ['AGN', 'QSO', 'Blazar', 'YSO'],
}


def _connection(path):
    db = sqlite3.connect(str(path), timeout=40)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA busy_timeout=40000')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS pages (
       subclass TEXT PRIMARY KEY, last_page INTEGER NOT NULL DEFAULT 0,
       exhausted INTEGER NOT NULL DEFAULT 0);
    CREATE TABLE IF NOT EXISTS candidates (
       oid TEXT NOT NULL, broad TEXT NOT NULL, subclass TEXT NOT NULL,
       query_rank INTEGER, hash_key TEXT NOT NULL, probability TEXT,
       PRIMARY KEY (oid, subclass));
    CREATE INDEX IF NOT EXISTS idx_candidate_class_order ON candidates(subclass,hash_key);
    CREATE INDEX IF NOT EXISTS idx_candidate_broad_order ON candidates(broad,hash_key);
    CREATE TABLE IF NOT EXISTS attempts (
       oid TEXT PRIMARY KEY, status TEXT NOT NULL, reason TEXT,
       subclass TEXT NOT NULL, timestamp TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS accepted (
       oid TEXT PRIMARY KEY, broad TEXT NOT NULL, subclass TEXT NOT NULL,
       hash_key TEXT NOT NULL, query_rank INTEGER, probability TEXT,
       photometry_file TEXT NOT NULL, timestamp TEXT NOT NULL, n_detections INTEGER NOT NULL);
    CREATE INDEX IF NOT EXISTS idx_accepted_class ON accepted(subclass);
    CREATE INDEX IF NOT EXISTS idx_accepted_broad ON accepted(broad);
    ''')
    db.commit()
    return db


def _set_signature(db, spec):
    signature=json.dumps(spec,sort_keys=True)
    old=db.execute("SELECT value FROM settings WHERE key='identity'").fetchone()
    if old and old[0]!=signature:
        raise ValueError('Collector checkpoint has a different seed, target, classifier, or page size. Use a NEW input folder to change those settings.')
    db.execute("INSERT OR REPLACE INTO settings VALUES ('identity',?)",(signature,))
    db.commit()


def _request(client, method, kwargs, delay, retries=3):
    last=None
    for attempt in range(retries):
        try:
            res=getattr(client,method)(**kwargs)
            if delay: time.sleep(delay)
            return pd.DataFrame(res)
        except Exception as exc:
            last=exc
            if attempt<retries-1:time.sleep(min(12.,2.**attempt))
    raise RuntimeError(f'ALeRCE {method} failed after {retries} attempts: {last}')


def _valid_curve(frame):
    if 'jd' in frame.columns and 'mjd' not in frame.columns:
        frame=frame.copy()
        frame['mjd']=pd.to_numeric(frame['jd'],errors='coerce')-2400000.5
    required={'mjd','magpsf','sigmapsf','fid'}
    if not required.issubset(frame.columns):
        raise ValueError('Required detections columns missing: '+str(sorted(required-set(frame.columns))))
    df=frame[['mjd','magpsf','sigmapsf','fid']].copy()
    for c in ('mjd','magpsf','sigmapsf'):df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['mjd','magpsf','sigmapsf'])
    df=df[(df.sigmapsf>0)&np.isfinite(df.sigmapsf)].sort_values('mjd')
    if len(df)<20 or df.mjd.nunique()<2:return None
    return df


def _export_manifest(db,folder,signature,target):
    rows=pd.read_sql_query('''SELECT oid AS object_id,broad AS broad_class,
        subclass AS original_class, 'ALeRCE broker_model pseudolabel' AS label_source,
        1 AS source_verified, photometry_file, timestamp AS retrieved_at,
        query_rank,probability AS broker_probability,n_detections AS raw_valid_detections
        FROM accepted ORDER BY oid''',db)
    rows['sampling_frame']='ALeRCE class-filtered pages sorted oid ASC; deterministic hash ranking within finite returned frame; NOT ZTF-stream random sample'
    rows['classifier']=signature['classifier'];rows['classifier_version']=signature['version']
    manifest=folder/'objects.csv'
    temp=manifest.with_suffix('.csv.tmp');rows.to_csv(temp,index=False);temp.replace(manifest)
    by_class=rows.broad_class.value_counts().to_dict() if len(rows) else {}
    by_subclass=rows.original_class.value_counts().to_dict() if len(rows) else {}
    pages={key:{'last_page':p,'exhausted':bool(e)} for key,p,e in db.execute('SELECT subclass,last_page,exhausted FROM pages')}
    statuses={k:int(v) for k,v in db.execute('SELECT status,COUNT(*) FROM attempts GROUP BY status')}
    n=int(len(rows));meta={
      'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'dataset_target':target,'actual_downloaded_eligible':n,
      'target_reached':n>=target,'shortfall':max(0,target-n),
      'achieved_broad_class_counts':by_class,'achieved_subclass_counts':by_subclass,
      'pages':pages,'candidate_ids':db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0],
      'attempts':statuses,'cross_broad_label_conflicts':db.execute("SELECT COUNT(*) FROM (SELECT oid FROM candidates GROUP BY oid HAVING COUNT(DISTINCT broad)>1)").fetchone()[0], 'query_source':'ALeRCE ZTF API','client_version':version_info().get('alerce'),
      'collector_settings':signature,
      'label_warning':'ALeRCE class predictions are pseudo-labels, not spectroscopic truth; do not claim independent astrophysical accuracy.',
      'sampling_warning':'Class-conditioned oid-sorted finite page frame is not population-probability sampling; do not claim ZTF stream representativeness.',
      'eligibility_warning':'Only objects with at least 20 valid detections and >=2 distinct times are included.',
      'completion_warning':'A larger collection target does not imply the broker can supply that many objects; report achieved counts.'}
    write_json(folder/'collection_metadata.json',meta)
    return meta


def collect(folder, per_subclass=None, pages=200, page_size=500, seed=42,
            classifier='lc_classifier',version='hierarchical_random_forest_1.0.0',
            target_objects=240000,delay=.20,max_queries=None,max_downloads=None,query_only=False,client=None):
    if per_subclass is not None:
        if per_subclass<1: raise ValueError('per_subclass must be positive')
        target_objects=int(per_subclass)*12
    if target_objects<1 or pages<1 or page_size<1:raise ValueError('target_objects,pages,page_size must be positive')
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    curves=folder/'lightcurves';curves.mkdir(exist_ok=True)
    spec={'seed':int(seed),'classifier':classifier,'version':version,
          'page_size':int(page_size),'target_objects':int(target_objects)}
    if client is None:
        try:from alerce.core import Alerce
        except ImportError as e:raise RuntimeError('Install pip install -r requirements-alerce.txt') from e
        client=Alerce()
    db=_connection(folder/'collection_progress.sqlite3')
    _set_signature(db,spec)
    target_sub=int(math.ceil(target_objects/12))
    target_broad=int(math.ceil(target_objects/3))
    query_count=0;downloads=0;query_interrupted=False;query_errors=[]
    # Persist *all* available pages up to cap before hash-based selection; this
    # means partial queries cannot bias the final selected subset unpredictably.
    for broad,subclasses in GROUPS.items():
        for subclass in subclasses:
            current=db.execute('SELECT last_page,exhausted FROM pages WHERE subclass=?',(subclass,)).fetchone()
            prev,exhausted=current if current else (0,0)
            if exhausted:continue
            for page in range(prev+1,pages+1):
                if max_queries is not None and query_count>=max_queries:
                    query_interrupted=True;break
                kwargs={'classifier':classifier,'class_name':subclass,'survey':'ztf',
                        'format':'pandas','page':page,'page_size':page_size,
                        'order_by':'oid','order_mode':'ASC'}
                try:data=_request(client,'query_objects',kwargs,delay)
                except Exception as e:
                    query_errors.append({'subclass':subclass,'page':page,'error':str(e)})
                    print(f'QUERY WARNING {subclass} page {page}: {e}',flush=True)
                    break
                query_count+=1
                idcol=next((v for v in ('oid','objectId','object_id') if v in data.columns),None)
                if len(data) and idcol is None:raise ValueError('ALeRCE object query lacks oid: '+str(data.columns.tolist()))
                selected=[]
                for rank,(_,r) in enumerate(data.iterrows()):
                    oid=str(r[idcol]);prob=r.get('probability',None)
                    selected.append((oid,broad,subclass,(page-1)*page_size+int(rank)+1,
                                     stable_key(oid,seed),None if pd.isna(prob) else str(prob)))
                with db:
                    db.executemany('INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?)',selected)
                    db.execute('INSERT INTO pages(subclass,last_page,exhausted) VALUES (?,?,?) ON CONFLICT(subclass) DO UPDATE SET last_page=excluded.last_page,exhausted=excluded.exhausted',
                              (subclass,page,int(len(data)<page_size)))
                if page==1 or page%20==0 or len(data)<page_size:
                    print(f'{subclass}: completed page {page}; got {len(data)} IDs in page; checkpoint saved',flush=True)
                if len(data)<page_size:break
            if query_interrupted:break
        if query_interrupted:break
    if not query_interrupted:
        # Do not mark capped pagination as fully exhaustive: pages may remain.
        print('Candidate page query phase complete to configured page cap.',flush=True)
    else:
        print('Paused candidate retrieval at --max-queries. Re-run same command to resume.',flush=True)
    if query_interrupted or query_only:
        meta=_export_manifest(db,folder,spec,target_objects)
        meta.update({'stage':'pages_checkpoint' if query_interrupted else 'pages_ready',
                     'query_failures_this_run':query_errors,'requested_page_cap':pages,'pages_this_run':query_count})
        write_json(folder/'collection_metadata.json',meta)
        db.close()
        return {'status':'PAUSED_QUERY' if query_interrupted else ('QUERY_ERRORS' if query_errors else 'PAGES_READY'),
                'downloaded':meta['actual_downloaded_eligible'],'manifest':str(folder/'objects.csv'),
                'query_failures':len(query_errors),'pages_this_run':query_count}

    def accept_candidate(row):
        nonlocal downloads
        oid,broad,subclass,rank,h,prob=row
        if db.execute('SELECT 1 FROM attempts WHERE oid=?',(oid,)).fetchone():return
        now=datetime.datetime.now(datetime.timezone.utc).isoformat()
        # Exclude objects with inconsistent pseudolabels across broad groups.
        # Never arbitrarily assign them to the first encountered class.
        conflict=db.execute('SELECT COUNT(DISTINCT broad) FROM candidates WHERE oid=?',(oid,)).fetchone()[0]
        if conflict>1:
            with db:db.execute('INSERT OR IGNORE INTO attempts VALUES (?,?,?,?,?)', (oid,'CONFLICT','inconsistent_broad_class_labels',subclass,now))
            downloads+=1
            return
        try:
            outfile=curves/(oid+'.csv')
            if outfile.exists():raw=_valid_curve(pd.read_csv(outfile))
            else:
                response=_request(client,'query_detections',{'oid':oid,'survey':'ztf','format':'pandas'},delay)
                raw=_valid_curve(response)
                if raw is not None:
                    part=outfile.with_suffix('.tmp');raw.to_csv(part,index=False);part.replace(outfile)
            if raw is None:
                status='INELIGIBLE';reason='fewer_than_20_valid_detections_or_missing_distinct_times'
            else:
                status='ACCEPTED';reason=None
            with db:
                db.execute('INSERT OR IGNORE INTO attempts VALUES (?,?,?,?,?)',(oid,status,reason,subclass,now))
                if status=='ACCEPTED':
                    db.execute('INSERT OR IGNORE INTO accepted VALUES (?,?,?,?,?,?,?,?,?)',
                               (oid,broad,subclass,h,rank,prob,str(outfile.relative_to(folder)),now,int(len(raw))))
        except Exception as e:
            # Failing a request is not permanently treated as an ineligible
            # astronomical object: it may succeed on the next invocation.
            print(f'WARNING {oid} retryable error: {e}',flush=True)
            return
        downloads+=1

    # Each pseudo-subclass gets first opportunity to fill 1/12 of the target.
    for broad,subclasses in GROUPS.items():
        for subclass in subclasses:
            got=db.execute('SELECT COUNT(*) FROM accepted WHERE subclass=?',(subclass,)).fetchone()[0]
            for row in db.execute('SELECT oid,broad,subclass,query_rank,hash_key,probability FROM candidates WHERE subclass=? ORDER BY hash_key',(subclass,)):
                if got>=target_sub:break
                if max_downloads is not None and downloads>=max_downloads:break
                was=db.execute('SELECT status FROM attempts WHERE oid=?',(row[0],)).fetchone()
                if was:continue
                accept_candidate(row)
                got=db.execute('SELECT COUNT(*) FROM accepted WHERE subclass=?',(subclass,)).fetchone()[0]
                if downloads%1000==0 and downloads>0:print(f'Download attempts {downloads}; {subclass} accepted {got}/{target_sub}',flush=True)
            if max_downloads is not None and downloads>=max_downloads:break
        if max_downloads is not None and downloads>=max_downloads:break
    # Fill class shortfalls from surplus candidates from other subclasses,
    # *without* silently violating the broad-class target.
    if max_downloads is None or downloads<max_downloads:
        for broad in GROUPS:
            got=db.execute('SELECT COUNT(*) FROM accepted WHERE broad=?',(broad,)).fetchone()[0]
            if got>=target_broad:continue
            for row in db.execute('SELECT oid,broad,subclass,query_rank,hash_key,probability FROM candidates WHERE broad=? ORDER BY hash_key',(broad,)):
                if got>=target_broad:break
                if max_downloads is not None and downloads>=max_downloads:break
                if db.execute('SELECT 1 FROM attempts WHERE oid=?',(row[0],)).fetchone():continue
                accept_candidate(row)
                got=db.execute('SELECT COUNT(*) FROM accepted WHERE broad=?',(broad,)).fetchone()[0]
                if downloads%1000==0 and downloads>0:print(f'Top-up attempts {downloads}; {broad} accepted {got}/{target_broad}',flush=True)
            if max_downloads is not None and downloads>=max_downloads:break
    meta=_export_manifest(db,folder,spec,target_objects);meta['query_failures_this_run']=query_errors
    write_json(folder/'collection_metadata.json',meta);db.close()
    state='TARGET_REACHED' if meta['target_reached'] else ('PAUSED_DOWNLOAD' if max_downloads is not None and downloads>=max_downloads else 'PARTIAL_COHORT')
    print(f"Collected eligible objects: {meta['actual_downloaded_eligible']}/{target_objects} (status={state}). Resume with same command.",flush=True)
    return {'status':state,'manifest':str(folder/'objects.csv'),'actual_eligible':meta['actual_downloaded_eligible'],
            'target':target_objects,'by_broad_class':meta['achieved_broad_class_counts'],
            'queries_this_run':query_count,'download_attempts_this_run':downloads}
