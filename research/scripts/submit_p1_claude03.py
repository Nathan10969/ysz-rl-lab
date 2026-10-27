"""Explicitly submit the 11 conditionally approved P1 jobs, once gate is satisfied."""
import datetime,hashlib,io,json,shlex,subprocess,sys,tarfile
from pathlib import Path
from claude03_gate import p1_gate
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'calibration'
def ssh(command,payload=b'',timeout=45):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','hzw-cpu',command],
                     input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    if r.returncode:raise RuntimeError(r.stderr.decode('utf8','replace')+'\n'+r.stdout.decode('utf8','replace'))
    return r

def main():
    analysis=json.loads((C/'P0_ANALYSIS.json').read_text());budget=json.loads((C/'BUDGET.json').read_text())
    gate=p1_gate(analysis,budget);print(json.dumps(gate,indent=2),flush=True)
    if '--submit' not in sys.argv:return
    assert gate['allowed'],'P1 is not released'
    review_status=json.loads((C/'REVIEW_STATUS.json').read_text())
    assert review_status['approval_reference']=='REVIEW_CLAUDE_03_P0.md'
    assert review_status.get('further_submissions_allowed'), 'Workflow is held for review'
    age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(analysis['analyzed_at_utc'])).total_seconds()
    assert 0<=age<300,'Refresh accounting and SCF analysis before submission'
    assert not (C/'P1_SUBMISSIONS.json').exists(),'P1 attempt ledger exists; reconcile instead of repeating submission'
    jobs=[j for j in json.loads((C/'CLAUDE03_JOB_LIST.json').read_text()) if j['phase']=='P1_small']
    assert len(jobs)==11 and sum(j['hard_core_hours'] for j in jobs)==176
    buf=io.BytesIO();hashlines=[]
    with tarfile.open(fileobj=buf,mode='w') as tar:
        for j in jobs:
            d=C/'execution_claude03'/j['name']
            for name,want in j['input_hashes'].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==want
            for f in d.iterdir():
                tar.add(f,arcname=j['name']+'/'+f.name,recursive=False)
                hashlines.append(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {j["name"]}/{f.name}')
        data=('\n'.join(hashlines)+'\n').encode();item=tarfile.TarInfo('CLAUDE03_P1_INPUTS.sha256');item.size=len(data);item.mtime=int(datetime.datetime.now().timestamp());tar.addfile(item,io.BytesIO(data))
    remote='/online1/ycsc_hzw/hd100371/Jony/runs/01_calib'
    command='set -eu; cd '+remote+'; test ! -e P1_SUBMISSIONS.json; '+ ' && '.join('test ! -e '+j['name'] for j in jobs)+'; tar -xf -; sha256sum -c CLAUDE03_P1_INPUTS.sha256'
    r=ssh(command,buf.getvalue());(C/'evidence/Claude03_P1_upload.txt').write_bytes(r.stdout)
    code=r'''import datetime,json,subprocess
from pathlib import Path
root=Path('/online1/ycsc_hzw/hd100371/Jony/runs/01_calib')
assert not (root/'P1_SUBMISSIONS.json').exists()
names=PLAN_NAMES
ledger={'approval_reference':'REVIEW_CLAUDE_03_P0.md','phase':'P1_small','hard_core_hours':176,'max_concurrent':2,'release_gate':RELEASE_GATE,'jobs':[]}
for i,name in enumerate(names):
    d=root/name;assert not (d/'submitted_job_id.txt').exists()
    pair_start=(i//2)*2
    deps=[ledger['jobs'][k]['slurm_job_id'] for k in range(pair_start-2,pair_start)] if pair_start else []
    command=['sbatch','--parsable']
    if deps:command+=['--dependency=afterok:'+':'.join(deps)]
    command+=[str(d/'sub.sh')]
    ledger['pending_submission']=name;(root/'P1_SUBMISSIONS.json').write_text(json.dumps(ledger,indent=2)+'\n')
    r=subprocess.run(command,cwd=d,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,encoding='utf8',errors='replace',check=True)
    jid=r.stdout.strip().split(';')[0];assert jid.isdigit()
    (d/'submitted_job_id.txt').write_text(jid+'\n')
    meta=json.loads((d/'job.json').read_text())
    ledger['jobs'].append({'name':name,'logical_job':meta['logical_job'],'phase':'P1_small','slurm_job_id':jid,'cores':32,'limit_hours':.5,'hard_core_hours':16,'dependencies_afterok':deps,'submitted_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    ledger.pop('pending_submission',None);(root/'P1_SUBMISSIONS.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(name,jid,flush=True)
print(json.dumps(ledger,indent=2))
'''
    code=code.replace('PLAN_NAMES',repr([j['name'] for j in jobs])).replace('RELEASE_GATE',repr(gate))
    (C/'evidence/submit_P1_Claude03_exact_eleven.py').write_text(code)
    r=ssh('python3 -',code.encode(),timeout=60)
    (C/'evidence/Claude03_P1_submission_stdout.txt').write_bytes(r.stdout)
    text=r.stdout.decode('utf8');ledger=json.loads(text[text.index('{'):]);(C/'P1_SUBMISSIONS.json').write_text(json.dumps(ledger,indent=2)+'\n')
    status=json.loads((C/'REVIEW_STATUS.json').read_text());status.update(p1_state='SUBMITTED',p1_submitted_jobs=11,new_cloud_jobs_submitted=len(json.loads((C/'P0_SUBMISSIONS.json').read_text())['jobs'])+11,p0_state='CLAUDE03_LARGE_P0_COMPLETE',p2_and_later='HELD_FOR_LARGE_P0_REVIEW');(C/'REVIEW_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    print(text)

if __name__=='__main__':main()
