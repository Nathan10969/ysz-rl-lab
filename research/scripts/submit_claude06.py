"""Submit each approved Claude06 input once; dependency lanes bound concurrency."""
import argparse,datetime,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from submit_claude05 import ssh
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'calibration';G=ROOT/'calibration_gpu'
REF='REVIEW_CLAUDE_06_release.md'

def submit(gpu):
    home=G if gpu else C;host='your-cluster' if gpu else 'hzw-cpu'
    jobs=json.loads((home/'CLAUDE06_JOB_LIST.json').read_text())
    assert len(jobs)==(25 if gpu else 20),'Prepare complete manifest first'
    assert json.loads((C/'REVIEW_STATUS.json').read_text())['approval_reference']==REF
    lanes=8 if gpu else 2
    assert all(j['gpus']==int(gpu) and j['cores']==(6 if gpu else 32) for j in jobs)
    total=sum(j['cap'] for j in jobs)+(0.07416666666666666 if gpu else 253.68)
    assert total<=(40 if gpu else 3000)
    ledger_path=home/'CLAUDE06_SUBMISSIONS.json';assert not ledger_path.exists(),'Existing batch: reconcile only, no bulk retry'
    remote_root=str(Path(jobs[0]['remote_directory']).parent)
    pre=r'''import json,os,subprocess
from pathlib import Path
root=Path(REMOTE_ROOT)
assert not (root/'CLAUDE06_SUBMISSIONS.json').exists(),'Batch ledger already exists'
assert not (root/'CLAUDE06_STOP.json').exists(),'Batch stop exists'
queue=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j|%T|%C|%b']).decode()
assert not any('ysz_' in line for line in queue.splitlines()),'Other active YSZ jobs; reconcile concurrency first'
for name in JOB_NAMES:assert not (root/name).exists(),'Remote input directory already exists: '+name
print(json.dumps({'queue':queue,'preflight_passed':True}))
'''.replace('REMOTE_ROOT',repr(remote_root)).replace('JOB_NAMES',repr([j['name'] for j in jobs]))
    evidence=home/'evidence';evidence.mkdir(exist_ok=True)
    r=ssh(host,'python3 -',pre.encode());(evidence/'Claude06_before_submission.json').write_bytes(r.stdout)
    buf=io.BytesIO();hashlines=[]
    with tarfile.open(fileobj=buf,mode='w') as tar:
        for job in jobs:
            d=home/'execution_claude06'/job['name']
            assert not (d/'submitted_job_id.txt').exists()
            for fn,want in job['input_hashes'].items():assert hashlib.sha256((d/fn).read_bytes()).hexdigest()==want
            for file in d.iterdir():
                assert file.is_file() and file.name not in ['POTCAR','submitted_job_id.txt']
                name=job['name']+'/'+file.name;tar.add(file,arcname=name,recursive=False)
                hashlines.append(hashlib.sha256(file.read_bytes()).hexdigest()+'  '+name)
        payload=('\n'.join(hashlines)+'\n').encode();info=tarfile.TarInfo('CLAUDE06_UPLOAD.sha256');info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
    r=ssh(host,'set -eu; cd '+remote_root+'; tar -xf -; sha256sum -c CLAUDE06_UPLOAD.sha256',buf.getvalue())
    (evidence/'Claude06_upload.txt').write_bytes(r.stdout+r.stderr)
    # One persistent intent record is written BEFORE each sbatch. A network loss
    # never causes another submission; recover the ledger and queue instead.
    code=r'''import datetime,json,os,subprocess
from pathlib import Path
root=Path(REMOTE_ROOT);names=JOB_NAMES;lanes=LANES;ledger_path=root/'CLAUDE06_SUBMISSIONS.json'
assert not ledger_path.exists()
ledger={'approval_reference':'REVIEW_CLAUDE_06_release.md','max_concurrent':lanes,'no_retries':True,'jobs':[]}
def persist():
    temp=ledger_path.with_suffix('.json.tmp');temp.write_text(json.dumps(ledger,indent=2)+'\n');temp.replace(ledger_path)
persist()
for index,name in enumerate(names):
    assert not (root/'CLAUDE06_STOP.json').exists(),'A running job failed; stop expanding batch'
    d=root/name;meta=json.loads((d/'job.json').read_text())
    assert not (d/'submitted_job_id.txt').exists()
    assert not any(j['name']==name for j in ledger['jobs'])
    assert 'pending_submission' not in ledger
    queue=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j|%T']).decode()
    assert not any(('ysz_'+name) in row.split('|')[1:2] for row in queue.splitlines()),'Name already queued'
    deps=[ledger['jobs'][index-lanes]['slurm_job_id']] if index>=lanes else []
    flags=['--dependency=afterok:'+':'.join(deps),'--kill-on-invalid-dep=yes'] if deps else []
    args=flags+[str(d/'sub.sh')]
    result=subprocess.run(['sbatch','--test-only']+args,cwd=d,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    print('TEST '+name+' '+result.stdout.decode()+' '+result.stderr.decode(),flush=True)
    if result.returncode:raise RuntimeError('Slurm preflight rejected; no automatic retry')
    ledger['pending_submission']=dict(name=name,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());persist()
    result=subprocess.run(['sbatch','--parsable']+args,cwd=d,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:raise RuntimeError('Submission rejected: '+result.stderr.decode())
    jid=result.stdout.decode().strip().split(';')[0];assert jid.isdigit()
    (d/'submitted_job_id.txt').write_text(jid+'\n')
    row=dict(name=name,config=meta['config'],phase=meta['phase'],kind=meta['kind'],
             slurm_job_id=jid,approval_reference='REVIEW_CLAUDE_06_release.md',
             cores=meta['cores'],gpus=meta['gpus'],cap=meta['cap'],limit_hours=meta['limit_hours'],
             dependencies_afterok=deps,lane=index%lanes,
             submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    ledger['jobs'].append(row);ledger.pop('pending_submission');persist()
    print('SUBMITTED '+name+' '+jid,flush=True)
'''.replace('REMOTE_ROOT',repr(remote_root)).replace('JOB_NAMES',repr([j['name'] for j in jobs])).replace('LANES',str(lanes))
    (evidence/'Claude06_exact_submission.py').write_text(code)
    try:
        r=ssh(host,'python3 -',code.encode(),timeout=180)
        (evidence/'Claude06_submission.txt').write_bytes(r.stdout+r.stderr)
        print(r.stdout.decode(),flush=True)
    finally:
        r=ssh(host,'cat '+remote_root+'/CLAUDE06_SUBMISSIONS.json')
        ledger=json.loads(r.stdout);ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
        for row in ledger['jobs']:
            (home/'execution_claude06'/row['name']/'submitted_job_id.txt').write_text(row['slurm_job_id']+'\n')
        status=json.loads((C/'REVIEW_STATUS.json').read_text())
        status.update(p0_state='CLAUDE06_RUNNING',updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      awaiting='Claude06outputs;P1P2P2b/crosssite/fullrelax;trainingheld')
        status['claude06_gpu_job_ids' if gpu else 'claude06_cpu_job_ids']=[j['slurm_job_id'] for j in ledger['jobs']]
        status['training_state']='HELD_AWAITING_CLAUDE06_REVIEW'
        (C/'REVIEW_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cpu',action='store_true');parser.add_argument('--gpu',action='store_true');parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    assert args.submit and args.cpu!=args.gpu,'Useexactly--cpuor--gpuand--submit'
    submit(args.gpu)
