#!/usr/bin/env python3
"""Resumable, class-interleaved bulk collector for an EXISTING AstroGuard SQLite cache.

Keeps the original candidate sampling frame, pseudo-labels, existing accepted
light curves and eligibility decisions. Does not change the original checkpoint
identity, 240K research target or original scripts. Targets a single 240K
eligible total without per-broad/per-subclass quotas. Each new call visits
up to --max-new previously unattempted candidate IDs.

CAUTION: class-filtered, ID-paginated ALeRCE frame is NOT a random sample
from the live ZTF stream. These labels are broker pseudolabels.
"""
import argparse
import datetime as dt
import json
import sqlite3
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from suite.collect import GROUPS, _connection, _export_manifest, _request, _valid_curve
from suite.common import write_json


def setup_queue(db):
    db.executescript('''
      CREATE TABLE IF NOT EXISTS bulk_queue (
        oid TEXT PRIMARY KEY,
        broad TEXT NOT NULL,
        subclass TEXT NOT NULL,
        hash_key TEXT NOT NULL,
        query_rank INTEGER,
        probability TEXT,
        status TEXT NOT NULL DEFAULT 'PENDING');
      CREATE INDEX IF NOT EXISTS ix_bulk_choose ON bulk_queue(broad,status,hash_key);
    ''')
    # Refresh the queue on EVERY invocation: page caps can be increased after
    # an initial 18K collection. INSERT OR IGNORE preserves prior attempt state
    # while admitting previously unseen, non-conflicting candidate IDs.
    with db:
        db.execute('''
          INSERT OR IGNORE INTO bulk_queue
                (oid,broad,subclass,hash_key,query_rank,probability,status)
          SELECT oid,broad,subclass,hash_key,query_rank,probability,'PENDING'
          FROM (
            SELECT c.oid,c.broad,c.subclass,c.hash_key,c.query_rank,c.probability,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.oid
                       ORDER BY CASE WHEN c.probability IS NULL THEN 1 ELSE 0 END,
                                CAST(c.probability AS REAL) DESC, c.subclass ASC
                   ) AS rn
            FROM candidates c
            JOIN (
               SELECT oid FROM candidates
               GROUP BY oid HAVING COUNT(DISTINCT broad)=1
            ) AS allowed ON allowed.oid=c.oid
          ) ranked WHERE rn=1
        ''')
        # A newly discovered conflicting label invalidates even an older
        # single-class queue entry. Never attempt it using that earlier label.
        db.execute('''
          UPDATE bulk_queue SET status='CONFLICT'
          WHERE oid IN (
            SELECT oid FROM candidates
            GROUP BY oid HAVING COUNT(DISTINCT broad)>1
          )
        ''')
    # Synchronize outcomes from prior bulk/pilot/original collection. This is
    # repeatable and doesn't delete or overwrite any accepted photometry.
    with db:
        db.execute("UPDATE bulk_queue SET status='DONE' WHERE oid IN (SELECT oid FROM attempts) AND status<>'DONE'")
    return int(db.execute('SELECT COUNT(*) FROM bulk_queue').fetchone()[0])


def bulk(input_folder='input',output_folder='RESULTS',max_new=5000,request_delay=.2,
         target=240000,retry_errors=False,client=None):
    folder=Path(input_folder).resolve()
    output=Path(output_folder).resolve()
    dbfile=folder/'collection_progress.sqlite3'
    if not dbfile.is_file():raise FileNotFoundError(f'Missing existing SQLite checkpoint: {dbfile}')
    if max_new<=0 or request_delay<0 or target<=0:raise ValueError('Require positive limits and nonnegative delay')
    output.mkdir(parents=True,exist_ok=True)
    curves=folder/'lightcurves';curves.mkdir(exist_ok=True)
    db=_connection(dbfile)
    identity_row=db.execute("SELECT value FROM settings WHERE key='identity'").fetchone()
    if not identity_row:raise RuntimeError('Missing original collector identity. Aborting to protect original files.')
    identity=json.loads(identity_row[0])
    if client is None:
        try:from alerce.core import Alerce
        except ImportError as exc:raise RuntimeError('Run python -m pip install -r requirements-alerce.txt') from exc
        client=Alerce()
    queue_size=setup_queue(db)
    if retry_errors:
        with db:db.execute("UPDATE bulk_queue SET status='PENDING' WHERE status='REQUEST_ERROR'")
    before_total=int(db.execute('SELECT COUNT(*) FROM accepted').fetchone()[0])
    new_counts=Counter();errors=[];new=0;per_broad_attempts=Counter()
    names=list(GROUPS)
    while new<max_new:
        if before_total+new_counts['ACCEPTED']>=target:break
        progressed=False
        for broad in names:
            if new>=max_new or before_total+new_counts['ACCEPTED']>=target:break
            row=db.execute('''SELECT oid,broad,subclass,hash_key,query_rank,probability
                              FROM bulk_queue WHERE broad=? AND status='PENDING'
                              ORDER BY hash_key LIMIT 1''',(broad,)).fetchone()
            if row is None:continue
            oid,b,sub,key,rank,prob=row
            # Recheck deduplication against original attempts and accepted table.
            if db.execute('SELECT 1 FROM attempts WHERE oid=?',(oid,)).fetchone():
                with db:db.execute("UPDATE bulk_queue SET status='DONE' WHERE oid=?",(oid,))
                continue
            stamp=dt.datetime.now(dt.timezone.utc).isoformat()
            try:
                outfile=curves/(oid+'.csv')
                if outfile.is_file():
                    clean=_valid_curve(pd.read_csv(outfile))
                else:
                    response=_request(client,'query_detections',
                                      {'oid':oid,'survey':'ztf','format':'pandas'},
                                      request_delay)
                    clean=_valid_curve(response)
                    if clean is not None:
                        tmp=outfile.with_suffix('.csv.tmp')
                        clean.to_csv(tmp,index=False)
                        tmp.replace(outfile)
                status='ACCEPTED' if clean is not None else 'INELIGIBLE'
                reason='' if clean is not None else 'fewer_than_20_valid_detections_or_missing_distinct_times'
            except Exception as exc:
                status='REQUEST_ERROR';reason=str(exc)[:300]
                if len(errors)<40:errors.append({'broad':b,'subclass':sub,'reason':reason})
            with db:
                db.execute('UPDATE bulk_queue SET status=? WHERE oid=?',('DONE' if status!='REQUEST_ERROR' else status,oid))
                if status!='REQUEST_ERROR':
                    db.execute('INSERT OR IGNORE INTO attempts VALUES (?,?,?,?,?)',
                               (oid,status,reason,sub,stamp))
                if status=='ACCEPTED':
                    db.execute('INSERT OR IGNORE INTO accepted VALUES (?,?,?,?,?,?,?,?,?)',
                               (oid,b,sub,key,rank,prob,str(outfile.relative_to(folder)),stamp,int(len(clean))))
            new+=1;per_broad_attempts[b]+=1;new_counts[status]+=1
            progressed=True
            if new%250==0:
                accepted=db.execute('SELECT COUNT(*) FROM accepted').fetchone()[0]
                print(f'Bulk attempts {new}/{max_new}; total eligible {accepted}/{target}; recent statuses {dict(new_counts)}',flush=True)
        if not progressed:break
    meta=_export_manifest(db,folder,identity,int(identity['target_objects']))
    left=dict(db.execute("SELECT broad,COUNT(*) FROM bulk_queue WHERE status='PENDING' GROUP BY broad"))
    statuscounts=dict(db.execute('SELECT status,COUNT(*) FROM bulk_queue GROUP BY status'))
    classes=dict(db.execute('SELECT broad,COUNT(*) FROM accepted GROUP BY broad'))
    summary={
       'time_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
       'dataset_goal':target,'eligible_total':meta['actual_downloaded_eligible'],
       'eligible_by_broad':classes,'bulk_queue_distinct_nonconflicting':queue_size,
       'remaining_unattempted_by_broad':left,
       'bulk_queue_status_counts':statuscounts,
       'this_run_new_download_attempts':new,
       'this_run_attempts_by_broad':dict(per_broad_attempts),
       'this_run_statuses':dict(new_counts),
       'request_error_examples':errors,
       'original_collector_settings_unchanged':identity,
       'sampling_warning':'Class-filtered, oid-paginated broker candidates. Do not claim representative ZTF prevalence.',
       'label_warning':'ALeRCE pseudo-labels are not independently verified classifications.',
       'completion_state': 'TARGET_REACHED' if meta['actual_downloaded_eligible']>=target else
                           ('RETRYABLE_ERRORS_ONLY' if not left and statuscounts.get('REQUEST_ERROR',0)>0 else
                            ('NO_MORE_CANDIDATES' if not left else 'PAUSED_BULK'))
    }
    write_json(output/'BULK_PROGRESS.json',summary)
    with zipfile.ZipFile(output/'AstroGuard_QFV_18K_Bulk_Collection_Summary.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in (output/'BULK_PROGRESS.json',folder/'collection_metadata.json'):
            if p.is_file():z.write(p,arcname=p.name)
    db.close()
    print('Bulk progress:',summary['completion_state'])
    print('Total accepted:',summary['eligible_total'],'By class:',classes)
    print('Unattempted by class:',left)
    print('Upload:',output/'AstroGuard_QFV_18K_Bulk_Collection_Summary.zip')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input',default='input',help='Original directory with existing SQLite checkpoint')
    p.add_argument('--output',default='RESULTS')
    p.add_argument('--max-new',type=int,default=5000,help='New candidate download attempts in this run')
    p.add_argument('--target',type=int,default=240000,help='Target TOTAL eligible objects; no class-specific quotas')
    p.add_argument('--request-delay',type=float,default=.2)
    p.add_argument('--retry-errors',action='store_true',help='Retry old network/API failures marked REQUEST_ERROR')
    a=p.parse_args()
    bulk(a.input,a.output,a.max_new,a.request_delay,a.target,a.retry_errors)

if __name__=='__main__':main()
