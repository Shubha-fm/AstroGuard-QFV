#!/usr/bin/env python3
"""Small reviewer benchmark, with a clean 18K target, no 240K checkpoint required.

Uses the frozen 18K manifest from run_18k_collect.py. Does NOT query
ALeRCE or download new data. Runs preparation, real model training, or
statistical analysis with the unchanged tested AstroGuard pipeline.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
from suite.common import write_json,sha256

MODELS={
    'classical':['rf','cnn','bigru','transformer','classical24a','classical24b'],
    'matched':['classical24a','classical24b','quantum24'],
    'smoke':['rf','classical24a','classical24b']
}
SEEDS=[42,142,242,342,442]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('stage',choices=['status','prepare','train','analyze','verify','latency','stress'])
    p.add_argument('--input',default='input',help='Existing folder containing frozen objects_18000.csv')
    p.add_argument('--output',default='RESULTS_18K')
    p.add_argument('--profile',choices=tuple(MODELS),default='matched')
    p.add_argument('--tag',default=None,help='Different tag for reruns with different epochs/prefixes')
    p.add_argument('--models',nargs='+',default=None)
    p.add_argument('--seeds',type=int,nargs='+',default=None)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--max-train',type=int,default=None,help='Train subset, e.g. 3000 for cost-matched quantum trial')
    p.add_argument('--bootstrap',type=int,default=3000)
    p.add_argument('--jar',default=None)
    p.add_argument('--per-class',type=int,default=6000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    a=p.parse_args(argv)
    base=Path(a.input).resolve();root=Path(a.output).resolve()
    manifest=base/f'objects_{a.per_class*3}.csv'
    freeze=base/f'objects_{a.per_class*3}_freeze.json'
    prepared=root/'prepared'
    profile=MODELS[a.profile] if a.models is None else a.models
    seeds=SEEDS if a.seeds is None else a.seeds
    if a.profile=='smoke':
        seeds=[42] if a.seeds is None else seeds
        if a.max_train is None:a.max_train=90
        if a.epochs==40:a.epochs=2
    elif a.profile=='matched' and a.max_train is None:
        # Match training-data budget across QML and both 24-weight controls.
        a.max_train=3000
    tag=a.tag or a.profile
    results=root/tag
    if a.stage=='status':
        print('Frozen manifest:',manifest,'exists:',manifest.is_file())
        if manifest.is_file():print('SHA256:',sha256(manifest))
        print('Expected:',a.per_class*3,'objects',a.per_class,'in each class')
        return
    if not manifest.exists() or not freeze.exists():
        raise FileNotFoundError('18K cohort not complete. First run run_18k_collect.py until READY_FOR_PREPARE.')
    import pandas as pd
    df=pd.read_csv(manifest,dtype={'object_id':str})
    if len(df)!=3*a.per_class or any(df.broad_class.value_counts().get(c,0)!=a.per_class for c in ('Periodic','Stochastic','Transient')):
        raise ValueError('Manifest not exactly balanced. Do not train until quota met.')
    if df.object_id.duplicated().any():raise ValueError('Duplicate object_ids in frozen manifest')
    freeze_data=json.loads(freeze.read_text(encoding='utf-8'))
    if sha256(manifest)!=freeze_data['sha256']:
        raise ValueError('Manifest changed after freezing. Do not train on an altered split.')
    root.mkdir(parents=True,exist_ok=True)
    protocol={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'cohort_manifest':str(manifest),'manifest_sha256':sha256(manifest),
              'objects':len(df),'split':'65/15/20 object-disjoint stratified',
              'class_balance':df.broad_class.value_counts().to_dict(),
              'label_truth':'ALeRCE classifier pseudolabels only',
              'sampling':'Class-filtered frame, not representative random ZTF alerts',
              'fresh_mac_18k_protocol':True,
              'profile':a.profile,'max_train':a.max_train,'models':profile,'seeds':seeds}
    write_json(root/'REVIEWER_18K_PROTOCOL.json',protocol)
    if a.stage=='prepare':
        from suite.preprocess_18k import prepare
        result=prepare(manifest,prepared,seed=a.seed)
        print(json.dumps(result,indent=2))
        return
    if not (prepared/'dataset_report.json').is_file():
        raise FileNotFoundError('Run: python run_18k.py prepare, before model training')
    # The original run_large.py reads the shared protocol.json for
    # reporting. Override only its in-memory 18K protocol, leaving the fresh collection
    # checkpoint identity and on-disk configuration untouched.
    import suite.common as common
    base_config=common.config
    def config_18k():
        c=base_config()
        c['split']={'train':.65,'val':.15,'test':.20}
        c['dataset_target']={'total_eligible_objects':a.per_class*3,
                             'target_per_broad_class':a.per_class,
                             'sampling_frame':'ALeRCE class-filtered pseudolabel cohort',
                             'achieved_size_must_be_reported':True}
        c['full']['expected_target_train_if_cohort_attained']=int(a.per_class*3*.65)
        c['full']['expected_target_validation_if_cohort_attained']=int(a.per_class*3*.15)
        c['full']['expected_target_test_if_cohort_attained']=int(a.per_class*3*.20)
        return c
    common.config=config_18k
    from run_large import main as original_main
    command=['--stage',a.stage,'--input',str(manifest),'--prepared',str(prepared),
             '--output',str(results),'--preset','full','--seed',str(a.seed),
             '--bootstrap',str(a.bootstrap),'--device',a.device,
             '--fail-fast']
    if a.stage in ('train','analyze'):
        command.extend(['--models',*profile,'--seeds',*map(str,seeds)])
    if a.stage=='train':
        command+=['--epochs',str(a.epochs),'--batch',str(a.batch),
                  '--objective','supervised','--prefixes','2','5','10','20','40','80','128',
                  '--include-predictions-in-zip']
        if a.max_train is not None:command+=['--max-train',str(a.max_train)]
    if a.stage=='verify' and a.jar:command+=['--jar',a.jar]
    original_main(command)

if __name__=='__main__':main()
