"""Runs real TLC, requiring exact properties for each mutation (not just any error)."""
import os,re,subprocess,time,sys
from pathlib import Path
from .common import ROOT,write_json,sha256
CASES=[('corrected','AstroGuardActor','AstroGuardActor.cfg',None),('gate','FaultGate','FaultGate.cfg','NoUnsafeRelease'),('stale','FaultStale','FaultStale.cfg','FreshRelease'),('retry','FaultRetry','FaultRetry.cfg','RetryInvalidatesAssessment'),('cross_object_isolation','FaultMixup','FaultMixupIsolation.cfg','ObjectIsolation')]

def verify(jar,out,timeout=1800):
    jar=Path(jar).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not jar.exists():raise FileNotFoundError(f'Missing real tla2tools.jar: {jar}')
    results=[];mdir=ROOT/'tla'
    for label,module,cfg,expected in CASES:
        cmd=['java','-cp',str(jar),'tlc2.TLC','-workers','1','-deadlock','-config',cfg,module+'.tla']
        t=time.perf_counter()
        try:r=subprocess.run(cmd,cwd=mdir,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf8',errors='replace',timeout=timeout,check=False)
        except subprocess.TimeoutExpired as e:
            (out/(label+'.log')).write_text(str(e),encoding='utf8');results.append({'case':label,'verdict':'TIMEOUT','expected_property':expected});continue
        log=r.stdout;(out/(label+'.log')).write_text('$ '+' '.join(cmd)+'\n'+log,encoding='utf8')
        errors=re.findall(r'Error:\s*(.+)',log)
        pass_ok= bool(re.search(r'Model checking completed\. No error has been found\.',log)) and r.returncode==0
        expected_failure= bool(expected and re.search(r'(?:Invariant|Action property|Temporal property|Property)\s+'+re.escape(expected)+r'\s+is violated',log,re.I))
        verdict='PASS' if expected is None and pass_ok else 'EXPECTED_FAULT_DETECTED' if expected_failure else 'NOT_CONFIRMED'
        values={}
        for key,pattern in [('generated',r'(\d+) states generated'),('distinct',r'(\d+) distinct states found')]:
            m=re.search(pattern,log);values[key]=int(m.group(1)) if m else None
        results.append({'case':label,'model':module,'cfg':cfg,'expected_property':expected,'verdict':verdict,'return_code':r.returncode,'elapsed_seconds':round(time.perf_counter()-t,3),'error_lines':errors[:3],**values})
        print(label,verdict,'distinct',values['distinct'],flush=True)
    data={'jar_sha256':sha256(jar),'cases':results,'all_expected':all(c['verdict'] in ('PASS','EXPECTED_FAULT_DETECTED') for c in results)}
    write_json(out/'tlc_summary.json',data)
    if not data['all_expected']:raise RuntimeError('Some TLC runs did not satisfy the expected property; examine TLC logs, do NOT report passing result')
    return data

def verify_scaling(jar,out,object_counts=(1,2,3),timeout=1800):
    """Bounded TLC scalability experiment. Every scale uses exactly its recorded cfg.
    Timeout is reported as timeout, never as a property PASS.
    """
    import shutil,tempfile
    jar=Path(jar).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not jar.is_file():raise FileNotFoundError(jar)
    results=[]
    with tempfile.TemporaryDirectory() as tmp:
        temp=Path(tmp);shutil.copy2(ROOT/'tla'/'AstroGuardActor.tla',temp/'AstroGuardActor.tla')
        for n in object_counts:
            if n<1 or n>5:raise ValueError('1..5 objects supported; larger state spaces may be intractable')
            cfg=f'''CONSTANTS
  Objects = {{{", ".join("o"+str(i) for i in range(1,n+1))}}}
  MaxVersion = 2
  MaxRetry = 1
SPECIFICATION Spec
INVARIANTS
  TypeOK
  NoUnsafeRelease
PROPERTIES
  FreshRelease
  RetryInvalidatesAssessment
  ObjectIsolation
'''
            (temp/'AstroGuardActor.cfg').write_text(cfg,encoding='utf8')
            (out/f'cfg_objects_{n}.cfg').write_text(cfg,encoding='utf8')
            cmd=['java','-cp',str(jar),'tlc2.TLC','-workers','1','-deadlock','-config','AstroGuardActor.cfg','AstroGuardActor.tla']
            t=time.perf_counter()
            try:
                completed=subprocess.run(cmd,cwd=temp,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf8',errors='replace',timeout=timeout,check=False)
                log=completed.stdout;returncode=completed.returncode
                verdict='PASS' if returncode==0 and 'Model checking completed. No error has been found.' in log else 'ERROR_OR_VIOLATION'
            except subprocess.TimeoutExpired as e:
                log='TIMEOUT after '+str(timeout)+' seconds\n'+str(e);verdict='TIMEOUT';returncode=None
            (out/f'tlc_objects_{n}.log').write_text(log,encoding='utf8')
            states=re.search(r'(\d+) distinct states found',log)
            generated=re.search(r'(\d+) states generated',log)
            results.append({'n_objects':n,'verdict':verdict,'returncode':returncode,'elapsed_s':round(time.perf_counter()-t,3),'distinct_states':int(states.group(1)) if states else None,'generated_states':int(generated.group(1)) if generated else None})
            print('TLC SCALING',n,'objects',verdict,results[-1]['distinct_states'],'states',flush=True)
    write_json(out/'tlc_scaling.json',{'runs':results,'nondeterministic_execution':'single-worker finite-state TLC; state counts dependent on specification/configuration'})
    return results
