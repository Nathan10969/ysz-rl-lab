"""Independent inspection of the released batch, allocations and dependency DAG."""
import datetime,hashlib,json,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
from ase.io import read
ROOT=Path(__file__).resolve().parents[1]
checks={};counts={}
for folder,expected,lanes in [('calibration',20,2),('calibration_gpu',25,8)]:
    home=ROOT/folder;jobs=json.loads((home/'CLAUDE06_JOB_LIST.json').read_text());counts[folder]=len(jobs)
    assert len(jobs)==expected and len({j['name'] for j in jobs})==expected
    for job in jobs:
        d=home/'execution_claude06'/job['name'];a=read(d/'POSCAR');small=job['n_atoms']==94
        assert a.get_chemical_symbols()==(['Zr']*28+['Y']*4+['O']*62 if small else ['Zr']*92+['Y']*16+['O']*208)
        assert np.allclose(a.cell.lengths(),10.28 if small else 15.42)
        original=read(ROOT/job['source_directory']/'POSCAR')
        assert np.allclose(a.positions,original.positions,atol=1e-10,rtol=0)
        inc={line.split('=',1)[0].strip():line.split('=',1)[1].strip() for line in (d/'INCAR').read_text().splitlines() if '=' in line}
        assert inc['ENCUT']==str(job['encut']) and inc['EDIFF']=='1E-5' and inc['AMIX']=='0.2' and inc['BMIX']=='0.0001'
        assert inc['LREAL']==job['lreal'] and int(inc['NSW'])==(200 if job['kind']=='relax' else 0)
        assert list(map(int,(d/'KPOINTS').read_text().splitlines()[3].split()))==job['kmesh']
        if job['kind']=='relax':assert inc['EDIFFG']=='-0.02' and inc['ISIF']=='2'
        if job['gpus']:
            assert job['gpus']==1 and job['cores']==6 and inc['NCORE']==inc['KPAR']=='1'
            assert job['binary']==('vasp_gam' if job['kmesh']==[1,1,1] else 'vasp_std')
            assert job['limit_seconds']==(14340 if job['kind']=='relax' else 900)
        else:assert job['cores']==32 and job['cap']==16 and job['limit_seconds']==1800
        for file,digest in job['input_hashes'].items():assert hashlib.sha256((d/file).read_bytes()).hexdigest()==digest
        for fn in ['run.sh','sub.sh']:subprocess.run(['/bin/bash','-n',str(d/fn)],check=True)
        batch=(d/'sub.sh').read_text();run=(d/'run.sh').read_text()
        assert '--no-requeue' in batch and 'CLAUDE06_STOP.json' in batch and 'timeout --signal=TERM --kill-after=10s' in batch
        assert 'check_result.py' in run and 'sha256sum -c REVIEWED_INPUTS.sha256' in run
    checks[folder+'_physical_inputs_and_scripts']=True
    ledger=home/'CLAUDE06_SUBMISSIONS.json'
    if ledger.exists():
        rows=json.loads(ledger.read_text())['jobs'];assert len(rows)==expected
        assert len({r['slurm_job_id'] for r in rows})==expected
        for index,row in enumerate(rows):
            assert row['name']==jobs[index]['name']
            assert (home/'execution_claude06'/row['name']/'submitted_job_id.txt').read_text().strip()==row['slurm_job_id']
            assert row['dependencies_afterok']==([rows[index-lanes]['slurm_job_id']] if index>=lanes else [])
            assert row['lane']==index%lanes
        checks[folder+'_one_attempt_each_dependency_limit_'+str(lanes)]=True
cpu=json.loads((ROOT/'calibration/CLAUDE06_JOB_LIST.json').read_text())
gpu=json.loads((ROOT/'calibration_gpu/CLAUDE06_JOB_LIST.json').read_text())
assert Counter(j['phase'] for j in cpu)=={'P1_small':11,'P1_lreal_pair':3,'P2_convergence':6}
assert Counter(j['phase'] for j in gpu)=={'P2b_maincell':16,'crosssite':4,'full_relax_cost':5}
assert 253.68+sum(j['cap'] for j in cpu)<=3000
assert .07416666666666666+sum(j['cap'] for j in gpu)<=40
checks['both_cumulative_caps_include_prior_failures']=True
status=json.loads((ROOT/'calibration/REVIEW_STATUS.json').read_text())
assert status['approval_reference']=='REVIEW_CLAUDE_06_release.md'
assert status.get('training_state','HELD_AWAITING_CLAUDE06_REVIEW')=='HELD_AWAITING_CLAUDE06_REVIEW'
checks['current_authority_and_training_held']=True
report=dict(verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,checks=checks,counts=counts,
            CPU_preexecution_full_cap=253.68+sum(j['cap'] for j in cpu),GPU_preexecution_full_cap=.07416666666666666+sum(j['cap'] for j in gpu))
(ROOT/'calibration/evidence/Claude06_full_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
