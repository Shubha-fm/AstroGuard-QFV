#!/usr/bin/env python3
"""Read-only safety check for a fresh ALeRCE candidate frame."""
import argparse
import json
import sqlite3
from pathlib import Path
from suite.collect import GROUPS


def frame_status(input_dir, pages=40):
    path=Path(input_dir)/'collection_progress.sqlite3'
    if not path.exists():
        return {'state':'NOT_INITIALIZED','database':str(path),'page_cap':pages,
                'message':'Run bash mac_run.sh query 100 40 to begin a fresh candidate search.'}
    db=sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True)
    try:
        signature=db.execute("SELECT value FROM settings WHERE key='identity'").fetchone()
        if not signature:
            return {'state':'UNEXPECTED_DB','message':'No collection identity in database'}
        ident=json.loads(signature[0]);summaries={};incomplete=[];total=0
        for broad,subs in GROUPS.items():
            for sub in subs:
                row=db.execute('SELECT last_page,exhausted FROM pages WHERE subclass=?',(sub,)).fetchone()
                page=int(row[0]) if row else 0; exhausted=bool(row[1]) if row else False
                n=int(db.execute('SELECT COUNT(*) FROM candidates WHERE subclass=?',(sub,)).fetchone()[0]);total+=n
                summaries[sub]={'broad':broad,'last_page':page,'exhausted':exhausted,'candidates':n}
                if not (exhausted or page>=pages):incomplete.append(sub)
        accepted=dict(db.execute('SELECT broad,COUNT(*) FROM accepted GROUP BY broad'))
        return {'state':'PAGES_READY' if not incomplete else 'QUERY_INCOMPLETE',
                'identity':ident,'page_cap':pages,'candidate_subclass_records':total,
                'accepted_by_broad':accepted,'incomplete_subclasses':incomplete,
                'pages':summaries,
                'message':('Run bash mac_run.sh collect 1000' if not incomplete
                           else 'Repeat bash mac_run.sh query 100 '+str(pages))}
    finally:db.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='input')
    p.add_argument('--pages',type=int,default=40);p.add_argument('--require-complete',action='store_true')
    a=p.parse_args();result=frame_status(a.input,a.pages)
    print(json.dumps(result,indent=2))
    if a.require_complete and result['state']!='PAGES_READY':
        raise SystemExit('Candidate query is not complete. '+result['message'])

if __name__=='__main__':main()
