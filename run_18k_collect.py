#!/usr/bin/env python3
"""18K balanced reviewer cohort. Uses a fresh checkpoint without changing its identity.

The fresh SQLite checkpoint and raw curves are preserved. A balanced
6,000-per-broad-class snapshot is frozen once all quotas are reached.
Class-filtered ALeRCE pseudolabels are NOT independent ground truth or a
probability sample of the natural ZTF alert stream.
"""
import argparse
import datetime as dt
import json
import sqlite3
import time
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from run_imbalanced_bulk import setup_queue
from suite.collect import _connection, _request, _valid_curve
from suite.common import stable_key, write_json, sha256

CLASSES = ('Transient', 'Periodic', 'Stochastic')


def counts(db):
    """Count only broad-class-UNAMBIGUOUS accepted sources."""
    return dict(db.execute('''
        SELECT a.broad, COUNT(*) FROM accepted a
        WHERE NOT EXISTS (
            SELECT 1 FROM candidates c WHERE c.oid=a.oid AND c.broad<>a.broad)
        GROUP BY a.broad
    '''))


def freeze_manifest(db, folder, per_class=6000, seed=42):
    """Once complete, write permanent object-disjoint cohort, never silently replace."""
    folder = Path(folder)
    target = len(CLASSES) * per_class
    by_class = counts(db)
    finished = all(by_class.get(c, 0) >= per_class for c in CLASSES)
    final = folder / f'objects_{target}.csv'
    temp = folder / f'objects_{target}_progress.csv'
    if final.exists():
        previous = pd.read_csv(final, dtype={'object_id':str})
        n = previous.broad_class.value_counts()
        if len(previous) != target or any(n.get(c, 0) != per_class for c in CLASSES):
            raise ValueError(f'Existing frozen cohort has unexpected class counts: {final}')
        return {'frozen':True,'path':str(final),'n':len(previous), 'sha256':sha256(final),
                'by_class':n.to_dict(),'warning':'Existing cohort reused unchanged'}

    rows = pd.read_sql_query('''
        SELECT a.oid AS object_id,a.broad AS broad_class,
               a.subclass AS original_class,
               'ALeRCE broker_model pseudolabel' AS label_source,
               1 AS source_verified,a.photometry_file,
               a.timestamp AS retrieved_at,a.query_rank,
               a.probability AS broker_probability,
               a.n_detections AS raw_valid_detections
        FROM accepted a
        WHERE NOT EXISTS (
            SELECT 1 FROM candidates c WHERE c.oid=a.oid AND c.broad<>a.broad)
    ''', db)
    # deterministic hash selection among eligible class-filtered candidates,
    # not random sampling from the ZTF alert stream.
    rows['sort_hash'] = rows.object_id.astype(str).map(lambda oid:stable_key(oid, seed))
    groups=[g.sort_values(['sort_hash','object_id']).head(per_class)
            for _,g in rows.groupby('broad_class')]
    if not groups:
        return {'frozen':False,'path':None,'n':0,'sha256':None,'by_class':{}}
    rows = pd.concat(groups, ignore_index=True)
    rows = rows.sort_values(['broad_class','sort_hash','object_id']).drop(columns='sort_hash')
    rows['sampling_frame']='ALeRCE class-filtered ID-paginated candidate frame; deterministic eligible-sample hash'
    rows['independent_label_verified'] = False
    rows['external_label_reference'] = ''
    rows['external_label'] = ''
    rows['source_verified'] = True  # data provenance, NOT astrophysical truth
    if rows.object_id.duplicated().any():
        raise RuntimeError('Cross-class object overlap after conflict exclusion')
    if finished:
        file = final
    else:
        file = temp
    tmp = file.with_suffix('.tmp')
    rows.to_csv(tmp, index=False)
    tmp.replace(file)
    if finished:
        # Exact cohort SHA is captured and should be cited alongside model results.
        write_json(folder / f'objects_{target}_freeze.json', {
            'file':file.name,'sha256':sha256(file),'seed':seed,
            'count':len(rows),'per_broad_class':per_class,
            'selection':'Stable hash of eligible IDs WITHIN ALeRCE class-filtered candidate frame',
            'label_warning':'ALeRCE predictions are pseudolabels; not independent confirmations',
            'time_utc':dt.datetime.now(dt.timezone.utc).isoformat()
        })
    return {'frozen':finished,'path':str(file),'n':len(rows),'sha256':sha256(file),
            'by_class':rows.broad_class.value_counts().to_dict()}


def collect_18k(input_dir='input', out_dir='RESULTS_18K', per_class=6000,
                max_new=3000, delay=.2, retry_errors=False, client=None,
                seed=42, only_status=False):
    folder=Path(input_dir).resolve()
    out=Path(out_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    db_file=folder/'collection_progress.sqlite3'
    if not db_file.is_file():
        raise FileNotFoundError(f'Fresh candidate checkpoint not found. Run mac_run.sh query: {db_file}')
    if per_class<1 or max_new<0 or delay<0:
        raise ValueError('Invalid --per-class, --max-new, or --request-delay')
    db=_connection(db_file)
    ident=db.execute("SELECT value FROM settings WHERE key='identity'").fetchone()
    if not ident:raise RuntimeError('Missing collection identity; refusing to create a new one')
    # Keep the fresh initial identity unchanged between resumed runs.
    original_identity=json.loads(ident[0])
    setup_queue(db)
    if retry_errors:
        with db:db.execute("UPDATE bulk_queue SET status='PENDING' WHERE status='REQUEST_ERROR'")
    completed=Counter(); errors=[]; attempted=Counter(); started=time.monotonic()
    live_count=Counter(counts(db))
    if client is None and not only_status and max_new>0:
        try:
            from alerce.core import Alerce
        except ImportError as exc:
            raise RuntimeError('Install: python -m pip install -r requirements-alerce.txt') from exc
        client=Alerce()
    if not only_status:
        folder.joinpath('lightcurves').mkdir(parents=True,exist_ok=True)
        broad_list=list(CLASSES)
        while sum(attempted.values())<max_new:
            needed=[c for c in broad_list if live_count[c]<per_class]
            if not needed:break
            made_progress=False
            # Larger deficits are attempted first, with equal turn-taking among
            # eligible groups. Once a group fills its quota it is skipped.
            for broad in sorted(needed,key=lambda c:(live_count[c]-per_class,broad_list.index(c))):
                if sum(attempted.values())>=max_new:break
                row=db.execute('''SELECT oid,subclass,hash_key,query_rank,probability
                                  FROM bulk_queue
                                  WHERE broad=? AND status='PENDING'
                                  ORDER BY hash_key LIMIT 1''',(broad,)).fetchone()
                if row is None:continue
                oid,sub,key,rank,prob=row
                if db.execute('SELECT 1 FROM attempts WHERE oid=?',(oid,)).fetchone():
                    with db:db.execute("UPDATE bulk_queue SET status='DONE' WHERE oid=?",(oid,))
                    made_progress=True
                    continue
                stamp=dt.datetime.now(dt.timezone.utc).isoformat()
                photo=folder/'lightcurves'/f'{oid}.csv'
                try:
                    if photo.exists(): raw=_valid_curve(pd.read_csv(photo))
                    else:
                        result=_request(client,'query_detections',
                            {'oid':oid,'survey':'ztf','format':'pandas'},delay)
                        raw=_valid_curve(result)
                        if raw is not None:
                            scratch=photo.with_suffix('.csv.tmp')
                            raw.to_csv(scratch,index=False)
                            scratch.replace(photo)
                    status='ACCEPTED' if raw is not None else 'INELIGIBLE'
                    reason='' if raw is not None else 'fewer_than_20_valid_detections_or_missing_times'
                except Exception as exc:
                    status='REQUEST_ERROR'
                    reason=str(exc)[:300]
                    if len(errors)<30:errors.append({'oid':oid,'message':reason})
                with db:
                    db.execute("UPDATE bulk_queue SET status=? WHERE oid=?",
                               ('REQUEST_ERROR' if status=='REQUEST_ERROR' else 'DONE',oid))
                    if status!='REQUEST_ERROR':
                        db.execute('INSERT OR IGNORE INTO attempts VALUES (?,?,?,?,?)',
                                   (oid,status,reason,sub,stamp))
                    if status=='ACCEPTED':
                        db.execute('INSERT OR IGNORE INTO accepted VALUES (?,?,?,?,?,?,?,?,?)',
                            (oid,broad,sub,key,rank,prob,str(photo.relative_to(folder)),stamp,int(len(raw))))
                if status=='ACCEPTED':live_count[broad]+=1
                attempted[broad]+=1
                completed[status]+=1
                made_progress=True
                n=sum(attempted.values())
                if n%250==0:
                    print(f'18K attempts {n}/{max_new}, accepted={dict(live_count)}, statuses={dict(completed)}', flush=True)
            if not made_progress:break
    total=counts(db)
    snap=freeze_manifest(db,folder,per_class,seed)
    remaining=dict(db.execute("SELECT broad,COUNT(*) FROM bulk_queue WHERE status='PENDING' GROUP BY broad"))
    ready=bool(snap['frozen'])
    summary={
        'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'target_balanced_total':per_class*3,
        'per_class_target':per_class,
        'accepted_unambiguous_by_class':total,
        'selected_manifest':snap,
        'new_attempts_by_class':dict(attempted),
        'new_statuses':dict(completed),
        'unattempted_candidates_by_class':remaining,
        'retryable_error_examples':errors,
        'candidate_frame_identity':original_identity,
        'elapsed_seconds':round(time.monotonic()-started,2),
        'completion':'READY_FOR_PREPARE' if ready else ('NO_MORE_CANDIDATES' if not remaining else 'PAUSED'),
        'sampling_warning':'Class-conditional broker selection is NOT population probability sampling',
        'label_warning':'Broker-predicted broad classes are NOT independently validated astronomical labels'
    }
    write_json(out/'COLLECTION_18K.json',summary)
    with zipfile.ZipFile(out/'AstroGuard_QFV_18K_Collection_Summary.zip','w',zipfile.ZIP_DEFLATED) as zipfile_out:
        zipfile_out.write(out/'COLLECTION_18K.json',arcname='COLLECTION_18K.json')
        freeze=folder/f'objects_{per_class*3}_freeze.json'
        if freeze.exists():zipfile_out.write(freeze,arcname=freeze.name)
    db.close()
    print('18K cohort:',summary['completion'], 'by class:',total)
    print('Manifest:',snap['path'],'rows=',snap['n'])
    print('Upload:',out/'AstroGuard_QFV_18K_Collection_Summary.zip')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='input',help='Fresh input directory containing the SQLite checkpoint')
    p.add_argument('--output',default='RESULTS_18K')
    p.add_argument('--per-class',type=int,default=6000)
    p.add_argument('--max-new',type=int,default=3000)
    p.add_argument('--request-delay',type=float,default=.2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--retry-errors',action='store_true')
    p.add_argument('--status',action='store_true',help='No network calls; only report progress and write manifest')
    a=p.parse_args()
    collect_18k(a.input,a.output,a.per_class,0 if a.status else a.max_new,
                a.request_delay,a.retry_errors,seed=a.seed,only_status=a.status)

if __name__=='__main__':main()
