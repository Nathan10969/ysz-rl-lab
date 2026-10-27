"""One-shot submission of the two user-approved GPU pilots; uncertain outcomes never retry."""
import datetime,hashlib,io,json,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];G=ROOT/'calibration_gpu'
REMOTE='/CLUSTER_HOME/runs/01_calib_gpu'

def ssh(command,payload=b'',timeout=60):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
                      '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','your-cluster',command],
                     input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    if r.returncode:raise RuntimeError(r.stdout.decode('utf8','replace')+'\n'+r.stderr.decode('utf8','replace'))
    return r

def main():
    jobs=json.loads((G/'JOB_LIST.json').read_text());budget=json.loads((G/'GPU_BUDGET.json').read_text())
    assert budget['authorized_card_hours']==3 and budget['actual_plus_remaining_caps']<=3
    assert [(j['name'],j['gpus']) for j in jobs]==[('07g_L01_1g',1),('08g_L01_2g',2)]
    assert sum(j['gpus']*j['limit_hours'] for j in jobs)==3
    assert not (G/'GPU_SUBMISSIONS.json').exists(),'Existing ledger: reconcile instead of repeating sbatch'
    buf=io.BytesIO();hashlines=[]
    with tarfile.open(fileobj=buf,mode='w') as tar:
        for j in jobs:
            d=G/'inputs'/j['name']
            for filename,want in j['input_hashes'].items():
                assert hashlib.sha256((d/filename).read_bytes()).hexdigest()==want
            for f in d.iterdir():
                assert f.name!='POTCAR' and f.is_file()
                tar.add(f,arcname=j['name']+'/'+f.name,recursive=False)
                hashlines.append(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {j["name"]}/{f.name}')
        for name,blob in [('GPU_INPUTS.sha256',('\n'.join(hashlines)+'\n').encode()),
                          ('GPU_BUDGET.json',(G/'GPU_BUDGET.json').read_bytes())]:
            item=tarfile.TarInfo(name);item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
    print(json.dumps({'inputs_verified':True,'jobs':[j['name'] for j in jobs],'authorized_card_hours':3,
                      'submission_requested':'--submit' in sys.argv}),flush=True)
    if '--submit' not in sys.argv:return
    guard=('test ! -e 07g_L01_1g/submitted_job_id.txt; test ! -e 08g_L01_2g/submitted_job_id.txt; '
           if '--resume-preflight' in sys.argv else 'test ! -e 07g_L01_1g; test ! -e 08g_L01_2g; ')
    r=ssh('set -eu; mkdir -p '+REMOTE+'; cd '+REMOTE+'; test ! -e GPU_SUBMISSIONS.json; '+
          guard+'tar -xf -; sha256sum -c GPU_INPUTS.sha256',buf.getvalue())
    (G/'evidence/upload.txt').write_bytes(r.stdout+r.stderr)
    code=r'''import datetime,json,os,subprocess
from pathlib import Path
root=Path('/CLUSTER_HOME/runs/01_calib_gpu')
assert not (root/'GPU_SUBMISSIONS.json').exists()
def run(args,cwd=None):
    r=subprocess.run(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     universal_newlines=True,encoding='utf8',errors='replace')
    if r.returncode:raise RuntimeError(str(args)+'\n'+r.stdout+'\n'+r.stderr)
    return r
q=run(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j|%T|%b']).stdout
assert not any('ysz_' in line for line in q.splitlines()),'Unexpected active project jobs: '+q
for name in ['07g_L01_1g','08g_L01_2g']:
    d=root/name;assert not (d/'submitted_job_id.txt').exists()
    r=run(['sbatch','--test-only',str(d/'sub.sh')],d)
    print('TEST_ONLY '+name+' '+r.stdout+' '+r.stderr,flush=True)
ledger={'authorization_reference':'GPU_PILOT_PLAN.md; user directly authorized in current chat',
        'authorized_card_hours':3,'max_concurrent_gpu_pilot_jobs':1,
        'dependency_design':'08g afterok07g; no auto retries; P1/P2 held','jobs':[]}
def persist():
    temp=root/'GPU_SUBMISSIONS.json.tmp';temp.write_text(json.dumps(ledger,indent=2)+'\n');temp.replace(root/'GPU_SUBMISSIONS.json')
for index,name in enumerate(['07g_L01_1g','08g_L01_2g']):
    d=root/name;assert not (d/'submitted_job_id.txt').exists()
    meta=json.loads((d/'job.json').read_text())
    deps=[ledger['jobs'][0]['slurm_job_id']] if index else []
    args=['sbatch','--parsable']
    if deps:args+=['--dependency=afterok:'+':'.join(deps)]
    args+=[str(d/'sub.sh')]
    ledger['pending_submission']=name;persist()
    r=run(args,d);jid=r.stdout.strip().split(';')[0];assert jid.isdigit(),'Uncertain sbatch result: no retry'
    (d/'submitted_job_id.txt').write_text(jid+'\n')
    ledger['jobs'].append({'name':name,'slurm_job_id':jid,'gpus':meta['gpus'],'mpi_ranks':meta['mpi_ranks'],
        'host_cpus':meta['host_cpus'],'limit_hours':1,'hard_card_hours':meta['gpus'],
        'dependencies_afterok':deps,'submitted_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    ledger.pop('pending_submission',None);persist()
    print('SUBMITTED '+name+' '+jid,flush=True)
print(json.dumps(ledger,indent=2),flush=True)
'''
    (G/'evidence/submission_exact_two.py').write_text(code)
    try:
        r=ssh('python3 -',code.encode(),timeout=90)
        (G/'evidence/submission_stdout.txt').write_bytes(r.stdout+r.stderr)
        print(r.stdout.decode('utf8','replace'),flush=True)
    finally:
        r=ssh('if test -f '+REMOTE+'/GPU_SUBMISSIONS.json; then cat '+REMOTE+'/GPU_SUBMISSIONS.json; fi')
        if not r.stdout.strip():
            raise RuntimeError('No paid submissions or durable ledger created: preflight failed; no automatic retry.')
        ledger=json.loads(r.stdout);(G/'GPU_SUBMISSIONS.json').write_text(json.dumps(ledger,indent=2)+'\n')
        for j in ledger['jobs']:(G/'inputs'/j['name']/'submitted_job_id.txt').write_text(j['slurm_job_id']+'\n')
        status_path=ROOT/'calibration/REVIEW_STATUS.json';status=json.loads(status_path.read_text())
        status.update(gpu_jobs_submitted=len(ledger['jobs']),gpu_authorized_card_hours=3,
                      gpu_authorization_reference='GPU_PILOT_PLAN.md',gpu_job_ids=[j['slurm_job_id'] for j in ledger['jobs']],
                      gpu_state='SUBMITTED' if len(ledger['jobs'])==2 and 'pending_submission' not in ledger else 'PARTIAL_OR_UNCERTAIN_STOP',
                      gpu_further_submissions_allowed=False,p1_state='HELD_CLAUDE04',p2_and_later='HELD')
        status_path.write_text(json.dumps(status,indent=2)+'\n')

if __name__=='__main__':main()
