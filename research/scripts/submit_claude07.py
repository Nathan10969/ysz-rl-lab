"""Submit only the nine reviewed Claude07 DFT jobs, once, with spending checks.

GPU resource lanes use afterany; scientific dependencies alone use afterok.
An ambiguous submission is an intent to reconcile, never permission to retry.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import shlex
import tarfile
import uuid

from submit_claude05 import ssh
from claude07_safety import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REF = 'REVIEW_CLAUDE_07_gates.md'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def submit(gpu):
    home = ROOT / ('calibration_gpu' if gpu else 'calibration') / 'claude07'
    jobs = json.loads((home / 'JOB_LIST.json').read_text())
    require(len(jobs) == (8 if gpu else 1), 'Exact reviewed manifest required')
    host = 'your-cluster' if gpu else 'hzw-cpu'
    old = json.loads((ROOT / ('calibration_gpu/GPU_BUDGET.json' if gpu else 'calibration/BUDGET.json')).read_text())
    baseline = 10.538055555555555 if gpu else 450.02666666666676
    actual = old['actual_card_hours' if gpu else 'cumulative_actual_core_hours']
    require(abs(actual-baseline) < 1e-7, 'Reconcile changed historical spending first')
    caps = sum(j['cap'] for j in jobs)
    require(abs(caps-(27.5 if gpu else 32)) < 1e-8, 'Unexpected round spending cap')
    require(baseline+caps <= (40 if gpu else 3000), 'Pool spending cap exceeded')
    require(all(j['gpus'] == int(gpu) and j['cores'] == (6 if gpu else 32) for j in jobs), 'Unexpected allocation')
    local_ledger = home / 'SUBMISSIONS.json'
    require(not local_ledger.exists(), 'Existing submission ledger: reconcile, never bulk retry')
    remote = str(Path(jobs[0]['remote_directory']).parent)
    require(all(str(Path(j['remote_directory']).parent) == remote for j in jobs), 'Remote roots disagree')
    claim_id = uuid.uuid4().hex
    evidence = home / 'evidence'
    evidence.mkdir(exist_ok=True, parents=True)
    pre = r'''import json,os,subprocess
from pathlib import Path
root=Path(REMOTE_ROOT)
def require(condition,message):
    if not condition:raise RuntimeError(message)
require(not (root/'SUBMISSIONS.json').exists(),'Remote ledger already exists; reconcile only')
require(not (root/'CLAUDE07_POOL_STOP.json').exists(),'Pool stop remains active')
q=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j|%T|%C|%b']).decode()
require(not any('ysz_' in x for x in q.splitlines()),'Existing YSZ work; reconcile concurrency first')
for name in NAMES:require(not (root/name).exists(),'Existing remote task directory: '+name)
root.mkdir(parents=True,exist_ok=True)
claim=root/'SUBMISSION_CLAIM.json'
fd=os.open(str(claim),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as handle:
    json.dump({'claim_id':CLAIM_ID,'approval_reference':'REVIEW_CLAUDE_07_gates.md'},handle)
    handle.flush();os.fsync(handle.fileno())
print(json.dumps({'queue':q,'preflight_passed':True}))
'''.replace('REMOTE_ROOT', repr(remote)).replace('NAMES', repr([j['name'] for j in jobs])).replace('CLAIM_ID', repr(claim_id))
    response = ssh(host, 'python3 -', pre.encode())
    (evidence / 'before_submission.json').write_bytes(response.stdout)
    archive = io.BytesIO()
    upload_hashes = []
    for job in jobs:
        d = home / 'execution' / job['name']
        require(not (d / 'submitted_job_id.txt').exists(), 'Task already has submission ID')
        for filename, digest in job['input_hashes'].items():
            require(hashlib.sha256((d / filename).read_bytes()).hexdigest() == digest, 'Input hash mismatch: '+job['name']+'/'+filename)
    with tarfile.open(fileobj=archive, mode='w') as tar:
        for job in jobs:
            d = home / 'execution' / job['name']
            for file in d.iterdir():
                require(file.is_file() and file.name not in ['POTCAR', 'submitted_job_id.txt'], 'Unexpected upload file: '+str(file))
                target = job['name'] + '/' + file.name
                tar.add(file, arcname=target, recursive=False)
                upload_hashes.append(hashlib.sha256(file.read_bytes()).hexdigest()+'  '+target)
        payload = ('\n'.join(upload_hashes)+'\n').encode()
        info = tarfile.TarInfo('UPLOAD.sha256'); info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        for file in [ROOT/REF, ROOT/'CLAUDE07_PREFLIGHT.md', home/'JOB_LIST.json']:
            tar.add(file, arcname=file.name, recursive=False)
    response = ssh(host, 'set -eu; cd '+shlex.quote(remote)+'; tar -xf -; sha256sum -c UPLOAD.sha256', archive.getvalue())
    (evidence / 'upload.txt').write_bytes(response.stdout+response.stderr)
    code = r'''import datetime,json,os,subprocess
from pathlib import Path
root=Path(REMOTE_ROOT);jobs=JOBS;path=root/'SUBMISSIONS.json'
def require(condition,message):
    if not condition:raise RuntimeError(message)
require(json.loads((root/'SUBMISSION_CLAIM.json').read_text())['claim_id']==CLAIM_ID,'Submission claim belongs to another runner')
require(not path.exists(),'Remote ledger already exists')
ledger={'approval_reference':'REVIEW_CLAUDE_07_gates.md','max_concurrent':2,'no_automatic_retries':True,'pool':POOL_NAME,'jobs':[]}
def persist():
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(ledger,indent=2)+'\n');tmp.replace(path)
persist()
for job in jobs:
    require(not (root/'CLAUDE07_POOL_STOP.json').exists(),'Budget/allocation pool stop')
    d=root/job['name'];require(not (d/'submitted_job_id.txt').exists(),'Existing task ID')
    require(not any(j['name']==job['name'] for j in ledger['jobs']),'Task already recorded')
    require('pending_submission' not in ledger,'Unreconciled submission intent')
    ids={j['name']:j['slurm_job_id'] for j in ledger['jobs']}
    flags=[];pieces=[]
    for key,mode in [('resource_afterany','afterany'),('depends_on','afterok')]:
        if job.get(key):pieces.append(mode+':'+':'.join(ids[name] for name in job[key]))
    if pieces:flags=['--dependency='+','.join(pieces),'--kill-on-invalid-dep=yes']
    q=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j|%T']).decode()
    require(not any(('ysz_'+job['name'])==x.split('|')[1] for x in q.splitlines()),'Name already queued')
    args=flags+[str(d/'sub.sh')]
    test=subprocess.run(['sbatch','--test-only']+args,cwd=str(d),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    print('TEST '+job['name']+' '+test.stdout.decode()+' '+test.stderr.decode(),flush=True)
    if test.returncode:raise RuntimeError('Slurm test-only rejected; no submission for this item')
    ledger['pending_submission']={'name':job['name'],'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};persist()
    result=subprocess.run(['sbatch','--parsable']+args,cwd=str(d),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:raise RuntimeError('Submission rejected or ambiguous: '+result.stderr.decode())
    jid=result.stdout.decode().strip().split(';')[0];require(jid.isdigit(),'Ambiguous Slurm ID; reconcile intent')
    (d/'submitted_job_id.txt').write_text(jid+'\n')
    row=dict(job,slurm_job_id=jid,dependencies_afterany=[ids[n] for n in job.get('resource_afterany',[])],
             dependencies_afterok=[ids[n] for n in job.get('depends_on',[])],
             submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    ledger['jobs'].append(row);ledger.pop('pending_submission');persist()
    print('SUBMITTED '+job['name']+' '+jid,flush=True)
'''.replace('REMOTE_ROOT', repr(remote)).replace('JOBS', repr(jobs)).replace('POOL_NAME', repr('DFT_GPU' if gpu else 'CPU')).replace('CLAIM_ID', repr(claim_id))
    (evidence / 'exact_submission.py').write_text(code)
    try:
        response = ssh(host, 'python3 -', code.encode(), timeout=180)
        (evidence / 'submission.txt').write_bytes(response.stdout+response.stderr)
        print(response.stdout.decode(), flush=True)
    finally:
        response = ssh(host, 'cat '+shlex.quote(remote+'/SUBMISSIONS.json'))
        ledger = json.loads(response.stdout)
        atomic_json(local_ledger, ledger)
        for row in ledger['jobs']:
            (home/'execution'/row['name']/'submitted_job_id.txt').write_text(row['slurm_job_id']+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true'); parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--submit', action='store_true'); args = parser.parse_args()
    if not args.submit or args.cpu == args.gpu:
        parser.error('Exactly --cpu or --gpu and --submit required')
    submit(args.gpu)
