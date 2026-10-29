#!/usr/bin/env python3
"""Evaluate independently verified labels against held-out model predictions.

Only user-supplied external confirmations are accepted, not ALeRCE class
predictions. This script cannot manufacture independent astronomical truth.
Input verified_labels.csv: object_id,verified_broad_class,catalog,reference,
verification_method (all required). Values Periodic/Stochastic/Transient.
"""
import argparse
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from suite.common import CLASSES,write_json

COLS=['object_id','verified_broad_class','catalog','reference','verification_method']

def evaluate(label_file,prediction_root,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    labels=pd.read_csv(label_file,dtype={'object_id':str}).fillna('')
    missing=set(COLS)-set(labels.columns)
    if missing:raise ValueError(f'Independent label file missing columns {sorted(missing)}')
    for col in COLS:
        if labels[col].astype(str).str.strip().eq('').any():raise ValueError(f'Blank {col}: provide external evidence, not ALeRCE labels')
    if labels.object_id.duplicated().any():raise ValueError('Duplicate independent object_id')
    if not labels.verified_broad_class.isin(CLASSES).all():raise ValueError('External class must be one of '+repr(CLASSES))
    if labels.catalog.astype(str).str.contains('alerce',case=False).any():
        raise ValueError('ALeRCE pseudolabels are not an independent external catalog')
    paths=sorted(Path(prediction_root).rglob('predictions.csv'))
    if not paths:raise FileNotFoundError('No model predictions: run training first')
    rows=[]; matrices=[]
    for path in paths:
        pred=pd.read_csv(path,dtype={'object_id':str})
        if not {'object_id','broad_class','prediction','split','prefix','model','seed'}.issubset(pred.columns):
            raise ValueError(f'Missing prediction columns in {path}')
        pred=pred[(pred.split=='test')&(pred.prefix==128)]
        joined=pred.merge(labels,on='object_id',validate='one_to_one')
        if joined.empty:continue
        true=joined.verified_broad_class.to_numpy()
        guessed=joined.prediction.to_numpy()
        broker=joined.broad_class.to_numpy()
        ok=np.isin(true,CLASSES)&np.isin(guessed,CLASSES)
        if not ok.all():raise ValueError('Unknown classes in verified prediction join')
        acc=float(accuracy_score(true,guessed))
        macro=float(f1_score(true,guessed,labels=CLASSES,average='macro',zero_division=0))
        rate=float(np.mean(true==broker))
        rows.append({'model':str(joined.model.iloc[0]),'seed':int(joined.seed.iloc[0]),
            'matched_independent_test_objects':len(joined),'independent_accuracy':acc,
            'independent_macro_f1':macro,'alerce_pseudolabel_agreement':rate,
            'warning':'External catalog must be checked for authoritative label confirmation and mismatched populations'})
        cm=confusion_matrix(true,guessed,labels=CLASSES)
        matrices.extend({'model':str(joined.model.iloc[0]),'seed':int(joined.seed.iloc[0]),
            'true':CLASSES[i],'predicted':CLASSES[j],'count':int(cm[i,j])} for i in range(3) for j in range(3))
    if not rows:
        raise ValueError('No independently labeled object IDs overlap the HELD-OUT TEST predictions; do not report external test accuracy')
    pd.DataFrame(rows).to_csv(out/'independent_accuracy.csv',index=False)
    pd.DataFrame(matrices).to_csv(out/'independent_confusion.csv',index=False)
    summary={'n_external_label_records':len(labels),'completed_model_seed_evaluations':len(rows),
        'limitation':'External label authority and class mapping require manual science review; matched labels may be class-selected'}
    write_json(out/'INDEPENDENT_LABELS_SUMMARY.json',summary)
    with zipfile.ZipFile(out/'AstroGuard_QFV_18K_Independent_Labels.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in ('INDEPENDENT_LABELS_SUMMARY.json','independent_accuracy.csv','independent_confusion.csv'):
            z.write(out/f,arcname=f)
    print(json.dumps(summary,indent=2)); print('Upload:',out/'AstroGuard_QFV_18K_Independent_Labels.zip')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',required=True)
    p.add_argument('--predictions',default='RESULTS_18K/matched/trained')
    p.add_argument('--output',default='RESULTS_18K/independent')
    x=p.parse_args()
    evaluate(x.labels,x.predictions,x.output)
