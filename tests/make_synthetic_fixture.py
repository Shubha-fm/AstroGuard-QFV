"""Generate ONLY for software testing. Must NEVER be used as astronomy evidence."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd

def build(base,n_each=18):
    base=Path(base);curves=base/'lightcurves';curves.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(9);rows=[]
    for j,label in enumerate(('Periodic','Stochastic','Transient')):
        for n in range(n_each):
            oid=f'SYNTHETIC_TEST_{j}_{n:04d}'
            t=np.arange(24,dtype=float)+rng.normal(0,.1,24)
            mag=(17+j)+({0:np.sin(t/3),1:rng.normal(0,.3,24),2:np.exp(-t/10)}[j])+rng.normal(0,.08,24)
            df=pd.DataFrame({'mjd':60000+t,'magpsf':mag,'sigmapsf':rng.uniform(.05,.2,24),'fid':np.where(np.arange(24)%2,1,2)})
            filename='lightcurves/'+oid+'.csv';df.to_csv(base/filename,index=False)
            rows.append({'object_id':oid,'broad_class':label,'original_class':'SYNTH_'+label,'label_source':'synthetic_fixture_NOT_REAL','source_verified':True,'photometry_file':filename,'sampling_frame':'explicit test fixture only; NOT ALeRCE'})
    pd.DataFrame(rows).to_csv(base/'objects.csv',index=False)
    (base/'NOT_ASTRONOMICAL_DATA.txt').write_text('SYNTHETIC SOFTWARE FIXTURE ONLY. DO NOT CITE ANY METRIC AS ASTRONOMICAL EVIDENCE.\n')
    print(base/'objects.csv')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',default='tests/tmp_fixture');a.add_argument('--n-each',type=int,default=18);b=a.parse_args();build(b.output,b.n_each)
