"""Submit the two approved fast jobs exactly once; never retries an uncertain sbatch."""
import datetime, hashlib, io, json, subprocess, sys, tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];C=ROOT/'calibration'
REMOTE='/online1/ycsc_hzw/hd100371/Jony/runs/01_calib'
REFERENCE='REVIEW_CLAUDE_04_L00.md'

def ssh(command,payload=b'',timeout=60):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
                      '-o','ConnectTimeout=15','-o','ServerAliveInterval=15',
                      '-o','ServerAliveCountMax=2','hzw-cpu',command],input=payload,
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    if r.returncode:raise RuntimeError(r.stdout.decode('utf8','replace')+'\n'+r.stderr.decode('utf8','replace'))
    return r

def main():
    jobs=json.loads((C/'CLAUDE04_JOB_LIST.json').read_text())
    budget=json.loads((C/'BUDGET.json').read_text())
    status=json.loads((C/'REVIEW_STATUS.json').read_text())
    assert status['approval_reference']==REFERENCE and status['p1_state']=='HELD_CLAUDE04'
    assert [j['name'] for j in jobs]==['05f_L00','06f_L01']
    assert sum(j['hard_core_hours'] for j in jobs)==256
    assert budget['actual_plus_remaining_caps']<=3000
    assert budget['p0_actual_plus_remaining_caps']<=216
    assert budget['remaining_fast_core_hour_caps']==256
    age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(budget['accounting_updated_at_utc'])).total_seconds()
    assert 0<=age<300,'Refresh accounting before submission'
    assert not (C/'FAST_SUBMISSIONS.json').exists(),'Submission ledger exists: reconcile, never submit again'
    buf=io.BytesIO();hashlines=[]
    with tarfile.open(fileobj=buf,mode='w') as tar:
        for j in jobs:
            d=C/'execution_claude04'/j['name']
            for filename,want in j['input_hashes'].items():
                assert hashlib.sha256((d/filename).read_bytes()).hexdigest()==want
            for f in sorted(d.iterdir()):
                assert f.is_file() and f.name!='POTCAR'
                tar.add(f,arcname=j['name']+'/'+f.name,recursive=False)
                hashlines.append(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {j["name"]}/{f.name}')
        data=('\n'.join(hashlines)+'\n').encode()
        info=tarfile.TarInfo('CLAUDE04_INPUTS.sha256');info.size=len(data)
        tar.addfile(info,io.BytesIO(data))
    print(json.dumps({'jobs':[j['name'] for j in jobs],'projected_all_calibration_core_hours':budget['actual_plus_remaining_caps'],
                      'fast_cap':256,'P1':'HELD','submission_requested':'--submit' in sys.argv}),flush=True)
    if '--submit' not in sys.argv:return
    r=ssh('set -eu; cd '+REMOTE+'; test ! -e FAST_SUBMISSIONS.json; test ! -e 05f_L00; test ! -e 06f_L01; '
          'tar -xf -; sha256sum -c CLAUDE04_INPUTS.sha256',buf.getvalue())
    (C/'evidence/Claude04_upload.txt').write_bytes(r.stdout)
    code=r'''import datetime,json,os,subprocess
from pathlib import Path
root=Path('/online1/ycsc_hzw/hd100371/Jony/runs/01_calib')
assert not (root/'FAST_SUBMISSIONS.json').exists(), 'Existing ledger: reconcile without retry'
assert not (root/'P1_SUBMISSIONS.json').exists(), 'Unexpected P1 submissions: stop and reconcile'
def run(args,cwd=None):
    r=subprocess.run(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     universal_newlines=True,encoding='utf8',errors='replace')
    if r.returncode:raise RuntimeError(str(args)+'\n'+r.stdout+'\n'+r.stderr)
    return r
queue=run(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j|%T|%C']).stdout
active=[line.split('|') for line in queue.splitlines() if '|ysz_' in line]
assert len(active)<=1 and all(row[0]=='1882470' for row in active), 'Unexpected active project jobs: '+queue
old_active=bool(active)
for name in ['05f_L00','06f_L01']:
    d=root/name
    assert not (d/'submitted_job_id.txt').exists()
    meta=json.loads((d/'job.json').read_text())
    assert meta['cores']==128 and meta['hard_core_hours']==128
    r=run(['sbatch','--test-only',str(d/'sub.sh')],d)
    print('TEST_ONLY '+name+' '+r.stdout+' '+r.stderr,flush=True)
ledger={'approval_reference':'REVIEW_CLAUDE_04_L00.md','phase':'CLAUDE04_fast',
        'hard_core_hours':256,'max_concurrent_project_jobs':2,'P1':'HELD','jobs':[],
        'old_L01_must_finish_naturally':True,'old_L01_active_at_submission':old_active,
        'dependency_design':'05f may overlap old04; 06f waits afterany old04 if still active; at most2 project jobs'}
ledger_path=root/'FAST_SUBMISSIONS.json'
def persist():
    temp=ledger_path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(ledger,indent=2)+'\n');temp.replace(ledger_path)
for index,name in enumerate(['05f_L00','06f_L01']):
    d=root/name;assert not (d/'submitted_job_id.txt').exists()
    cmd=['sbatch','--parsable']
    deps=['1882470'] if old_active and index==1 else []
    if deps:cmd+=['--dependency=afterany:'+':'.join(deps)]
    cmd+=[str(d/'sub.sh')]
    ledger['pending_submission']=name;persist()
    r=run(cmd,d)
    jid=r.stdout.strip().split(';')[0];assert jid.isdigit(), 'Uncertain submission outcome: do not retry'
    (d/'submitted_job_id.txt').write_text(jid+'\n')
    meta=json.loads((d/'job.json').read_text())
    ledger['jobs'].append({'name':name,'logical_job':meta['logical_job'],'config':meta['config'],
        'phase':'CLAUDE04_fast','slurm_job_id':jid,'cores':128,'limit_hours':1,'hard_core_hours':128,
        'approval_reference':'REVIEW_CLAUDE_04_L00.md','dependencies_afterany':deps,
        'submitted_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    ledger.pop('pending_submission',None);persist()
    print('SUBMITTED '+name+' '+jid,flush=True)
print(json.dumps(ledger,indent=2),flush=True)
'''
    (C/'evidence/submit_fast_Claude04_exact_two.py').write_text(code)
    try:
        r=ssh('python3 -',code.encode(),timeout=90)
        (C/'evidence/Claude04_submission_stdout.txt').write_bytes(r.stdout)
        print(r.stdout.decode('utf8','replace'),flush=True)
    finally:
        # Recover durable server records even if sbatch/SSH has an uncertain outcome.
        r=ssh('cat '+REMOTE+'/FAST_SUBMISSIONS.json')
        ledger=json.loads(r.stdout)
        (C/'FAST_SUBMISSIONS.json').write_text(json.dumps(ledger,indent=2)+'\n')
        for job in ledger['jobs']:
            (C/'execution_claude04'/job['name']/'submitted_job_id.txt').write_text(job['slurm_job_id']+'\n')
        status.update(fast_state='SUBMITTED' if len(ledger['jobs'])==2 and 'pending_submission' not in ledger else 'PARTIAL_OR_UNCERTAIN_STOP',
                      fast_job_ids=[j['slurm_job_id'] for j in ledger['jobs']],
                      new_cloud_jobs_submitted=len(json.loads((C/'P0_SUBMISSIONS.json').read_text())['jobs'])+len(ledger['jobs']),
                      further_submissions_allowed=False,awaiting='Fast results; no additional submissions authorized',
                      updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (C/'REVIEW_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')

if __name__=='__main__':main()
