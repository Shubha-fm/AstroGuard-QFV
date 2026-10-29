#!/usr/bin/env python3
"""AstroGuard extended experimental suite. Never fabricates astronomical evidence."""
import argparse,datetime,hashlib,json,os,platform,subprocess,sys,time,traceback,zipfile
from pathlib import Path
from suite.common import ROOT,config,write_json,sha256,version_info

def results_zip(output,include_predictions=False):
    """Collect evidence without silently bundling large raw light curves or weights."""
    output=Path(output)
    tag={'classical':'Classical', 'matched':'Matched', 'smoke':'Smoke',
         'quantum_sanity':'Quantum_Sanity'}.get(output.name,output.name)
    z=output/f'AstroGuard_QFV_18K_{tag}_Results.zip'
    omitted=[];included=[]
    files=sorted(f for f in output.rglob('*') if f.is_file() and f!=z)
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for file in files:
            rel=file.relative_to(output);name=str(rel).replace("\\",'/')
            if file.suffix.lower() in ('.zip','.pt','.joblib','.npz','.npy','.pkl','.sqlite3','.sqlite3-wal','.sqlite3-shm') or '/curves/' in '/'+name or name.startswith('prepared/curves/'):
                omitted.append({'file':name,'reason':'raw data or model checkpoint not included in evidence ZIP'});continue
            if file.name=='predictions.csv' and not include_predictions:
                omitted.append({'file':name,'reason':'large per-model individual predictions withheld; share separately if needed'});continue
            if file.stat().st_size>30_000_000:
                omitted.append({'file':name,'reason':'over 30 MB size limit; share separately if needed'});continue
            archive.write(file,arcname=name);included.append(name)
        archive.writestr('UPLOAD_MANIFEST.json',json.dumps({'included':included,'omitted':omitted,'note':'Large per-object data and model checkpoints are NOT included by default; preserve locally and share explicitly if requested'},indent=2))
    return z

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--stage',choices=['collect','prepare','train','analyze','stress','verify','latency','scale','tlc-scale','all'],required=True)
    ap.add_argument('--input',default='input/objects.csv',help='Manifest: object_id, broad_class, original_class, label_source, photometry_file, sampling_frame')
    ap.add_argument('--prepared',default='RESULTS/prepared',help='Prepared data folder (cache/splits)')
    ap.add_argument('--output',default='RESULTS',help='All output files; ZIP here')
    ap.add_argument('--preset',choices=['pilot','full'],default='pilot')
    ap.add_argument('--seed',type=int,default=42,help='Dataset selection + split seed')
    ap.add_argument('--seeds',nargs='+',type=int,help='Model seeds, override preset')
    ap.add_argument('--models',nargs='+',help='Model names, override preset; use --models quantum24 to select PennyLane branch')
    ap.add_argument('--epochs',type=int,help='Override training epochs')
    ap.add_argument('--objective',choices=['supervised','multitask'],help='Loss objective. Pilot defaults supervised, full defaults multitask')
    ap.add_argument('--batch',type=int,help='Override batch size')
    ap.add_argument('--max-train',type=int,default=None,help='Subset only TRAIN data; tests/val remain fixed')
    ap.add_argument('--limit',type=int,help='Prepare up to N provided objects via hash-based fixed sampling')
    ap.add_argument('--prefixes',nargs='+',type=int,help='Prefix lengths for retrospective early alert results')
    ap.add_argument('--jar',help='Path to official tla2tools.jar for TLC')
    ap.add_argument('--bootstrap',type=int,default=3000)
    ap.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    ap.add_argument('--timeout',type=int,default=1800,help='TLC timeout per module in seconds')
    ap.add_argument('--target-objects',type=int,default=18000,help='Total eligible pseudo-labeled objects targeted across 12 ALeRCE subclasses')
    ap.add_argument('--per-subclass',type=int,default=None,help='Small pilot override, replaces --target-objects with 12 x N')
    ap.add_argument('--max-queries',type=int,default=None,help='Maximum ALeRCE candidate-pages queried this invocation; re-run to resume')
    ap.add_argument('--max-downloads',type=int,default=None,help='Maximum light curves processed this invocation; re-run to resume')
    ap.add_argument('--query-only',action='store_true',help='Only build the checkpointed ID candidate pool; do NOT download photometry')
    ap.add_argument('--request-delay',type=float,default=0.20,help='Delay between successful broker requests in seconds')
    ap.add_argument('--pages',type=int,default=200,help='Maximum candidate pages per pseudo-subclass, checkpointed in SQLite')
    ap.add_argument('--page-size',type=int,default=500)
    ap.add_argument('--classifier',default='lc_classifier')
    ap.add_argument('--classifier-version',default='hierarchical_random_forest_1.0.0')
    ap.add_argument('--skip-stress',action='store_true')
    ap.add_argument('--latency-objects',type=int,default=100,help='Held-out single-object measured local end-to-end latency sample size')
    ap.add_argument('--scales',type=int,nargs='+',default=[1000,5000,20000,60000,120000,168000],help='Different TRAIN subset sizes; validation/test fixed')
    ap.add_argument('--tlc-objects',type=int,nargs='+',default=[1,2,3],help='Bounded TLC state-space scaling')
    ap.add_argument('--fail-fast',action='store_true')
    ap.add_argument('--include-predictions-in-zip',action='store_true',help='Include individual heldout per-model predictions.csv files in ZIP when under 30 MB each')
    args=ap.parse_args(argv)
    if args.bootstrap<100:ap.error('--bootstrap must be at least 100')
    if args.target_objects<1 or (args.max_queries is not None and args.max_queries<1) or (args.max_downloads is not None and args.max_downloads<1):ap.error('Collector target and non-null per-invocation limits must be positive')
    conf=config();opts=conf[args.preset]; output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    prepared=Path(args.prepared).resolve(); inputfile=Path(args.input).resolve()
    models=args.models or opts['models'];seeds=args.seeds or opts['seeds'];prefixes=args.prefixes or conf['prefixes']
    epochs=args.epochs or opts['epochs'];batch=args.batch or opts['batch_size'];objective=args.objective or ('multitask' if args.preset=='full' else 'supervised')
    if args.preset=='pilot' and args.max_train is None:max_train=opts['max_train']
    else:max_train=args.max_train
    print('AstroGuard extended experiment',datetime.datetime.now(datetime.timezone.utc).isoformat(),'\nStage:',args.stage,'Preset:',args.preset,'Results:',output,flush=True)
    summary={'planned_dataset_target':conf.get('dataset_target'), 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':args.stage,'preset':args.preset,'paths':{'input':str(inputfile),'prepared':str(prepared),'output':str(output)},'configuration':vars(args),'versions':version_info(),'steps':{},'caution':'Scientific outcomes valid only if real archived photometry and labels are provided. Unit test fixtures are NOT astronomy observations.'}
    def runstep(name,func):
        print('\n==========',name,'==========',flush=True);t=time.perf_counter()
        try:
            res=func();summary['steps'][name]={'status':'OK','seconds':round(time.perf_counter()-t,2),'result':res};print('Completed',name,flush=True)
        except Exception as e:
            tb=traceback.format_exc();(output/(name+'_ERROR.log')).write_text(tb,encoding='utf8')
            summary['steps'][name]={'status':'ERROR','seconds':round(time.perf_counter()-t,2),'message':str(e)}
            print('FAILED:',e,'\nSee:',output/(name+'_ERROR.log'),flush=True)
            if args.fail_fast:raise
        finally:
            write_json(output/'RUN_SUMMARY.json',summary)
    if args.stage=='collect':
        from suite.collect import collect
        def collecting():
            import shutil
            result=collect(inputfile.parent,per_subclass=args.per_subclass,pages=args.pages,page_size=args.page_size,seed=args.seed,classifier=args.classifier,version=args.classifier_version,target_objects=args.target_objects,delay=args.request_delay,max_queries=args.max_queries,max_downloads=args.max_downloads,query_only=args.query_only)
            source=inputfile.parent/'collection_metadata.json'
            if source.exists():
                shutil.copy2(source,output/'collection_metadata.json')
            return result
        runstep('collect',collecting)
    if args.stage in ['prepare','all']:
        from suite.preprocess import prepare
        runstep('prepare',lambda:prepare(inputfile,prepared,limit=args.limit,seed=args.seed))
    if args.stage in ['train','all']:
        from suite.train import train_model,MODELS
        if any(m not in MODELS for m in models):ap.error('Unknown --models: '+str([m for m in models if m not in MODELS]))
        def training():
            from suite.preprocess import CurveStore
            ds=CurveStore(prepared)
            reports=[]
            for m in models:
                for seed in seeds:
                    dest=output/'trained'/m/('seed_'+str(seed))
                    # Resume successful existing runs rather than overwriting.
                    if (dest/'predictions.csv').exists() and (dest/'train_summary.json').exists():
                        old=json.loads((dest/'train_summary.json').read_text())
                        import hashlib
                        curr_sha=hashlib.sha256(ds.df[['object_id','split','broad_class']].sort_values('object_id').to_csv(index=False).encode()).hexdigest()
                        now_setup={'epochs':epochs,'batch':batch,'hidden':opts['hidden'],'layers':opts['transformer_layers'],'heads':opts['attention_heads'],'max_train':max_train,'prefixes':list(prefixes),'objective':objective}
                        if old.get('data_manifest_sha256')!=curr_sha or old.get('setup')!=now_setup:
                            raise ValueError(f'Existing output for {m}/{seed} is from a DIFFERENT dataset or configuration. Use a new --output folder, do not silently mix runs.')
                        print('RESUME: matching complete run',dest,flush=True)
                        reports.append(old);continue
                    try:
                        rep=train_model(ds,m,seed,dest,epochs=epochs,batch=batch,hidden=opts['hidden'],layers=opts['transformer_layers'],heads=opts['attention_heads'],max_train=max_train,device=args.device,prefixes=prefixes,objective=objective)
                        reports.append(rep)
                    except Exception as e:
                        trace=traceback.format_exc();dest.mkdir(parents=True,exist_ok=True);(dest/'ERROR.log').write_text(trace,encoding='utf8')
                        print(f'ERROR training {m} seed {seed}: {e}',flush=True)
                        if args.fail_fast:raise
            write_json(output/'trained'/'all_runs.json',reports)
            if not reports:raise RuntimeError('No training experiments completed successfully')
            return {'complete_runs':len(reports),'models':models,'seeds':seeds,'max_train':max_train,'epochs':epochs}
        runstep('train',training)
    if args.stage in ['analyze','all']:
        from suite.analyze_large import run_analysis_large as run_analysis
        runstep('analyze',lambda:run_analysis(output/'trained',output/'analysis',bootstrap=args.bootstrap))
    if args.stage in ['stress','all'] and not args.skip_stress:
        from suite.stress import stress
        runstep('stress',lambda:stress(output/'trained',prepared,output/'stress',seed=args.seed,batch=batch))
    if args.stage=='scale':
        from suite.train import train_model
        from suite.preprocess import CurveStore
        ds=CurveStore(prepared)
        n=len(ds.indices('train'))
        for scale in args.scales:
            if scale>n:
                print('SKIP scale',scale,'greater than available train set',n,flush=True);continue
            for model in models:
                for sd in seeds:
                    runstep(f'scale_{scale}_{model}_{sd}',lambda scale=scale,model=model,sd=sd:train_model(ds,model,sd,output/'scalability'/f'train_{scale}'/'trained'/model/f'seed_{sd}',epochs=epochs,batch=batch,hidden=opts['hidden'],layers=opts['transformer_layers'],heads=opts['attention_heads'],max_train=scale,device=args.device,prefixes=[128],objective=objective))
            from suite.analyze_large import run_analysis_large as run_analysis
            runstep(f'scale_analysis_{scale}',lambda scale=scale:run_analysis(output/'scalability'/f'train_{scale}'/'trained',output/'scalability'/f'train_{scale}'/'analysis',bootstrap=args.bootstrap))
    if args.stage=='tlc-scale':
        from suite.verify import verify_scaling
        jar=args.jar or os.environ.get('TLA2TOOLS_JAR') or (str(ROOT/'tla2tools.jar') if (ROOT/'tla2tools.jar').is_file() else None)
        if not jar:raise FileNotFoundError('Use --jar path/to/tla2tools.jar for actual TLC scaling')
        runstep('tlc_scale',lambda:verify_scaling(jar,output/'tlc_scale',args.tlc_objects,args.timeout))
    if args.stage in ['latency','all']:
        from suite.latency import bench
        runstep('latency',lambda:bench(output/'trained',prepared,output/'latency',count=args.latency_objects,seed=args.seed))
    if args.stage in ['verify','all']:
        from suite.verify import verify
        jar=args.jar or os.environ.get('TLA2TOOLS_JAR') or (str(ROOT/'tla2tools.jar') if (ROOT/'tla2tools.jar').is_file() else None)
        if jar:runstep('verify',lambda:verify(jar,output/'tlc',args.timeout))
        else:
            summary['steps']['verify']={'status':'NOT_RUN','message':'Provide --jar PATH_TO_tla2tools.jar; did not invent formal results'}
            print('TLC skipped: no official tla2tools.jar supplied',flush=True)
    # Record protocol and source code hashes; never bundle original raw photometry by default.
    sources=list((ROOT/'suite').glob('*.py'))+list((ROOT/'tla').glob('*.tla'))+list((ROOT/'tla').glob('*.cfg'))+[ROOT/'configs'/'protocol.json']
    summary['source_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in sources}
    summary['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(output/'RUN_SUMMARY.json',summary)
    z=results_zip(output,include_predictions=args.include_predictions_in_zip)
    print('\nDONE. Upload:',z,'\nSize:',round(z.stat().st_size/1048576,2),'MB. ALSO upload the training logs for any errors.',flush=True)
    return summary

if __name__=='__main__':main()
