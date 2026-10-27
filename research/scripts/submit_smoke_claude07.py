"""One-shot submission of the two approved smoke jobs, behind both DFT lanes.

An exclusive claim plus a durable intent protects ambiguous sbatch outcomes.
This script never retries an existing claim or submission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import uuid

from claude07_safety import atomic_json
from submit_claude05 import ssh

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / 'training_smoke'
DFT_REMOTE = '/CLUSTER_HOME/runs/02_calib_c07'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked_inputs():
    manifest = json.loads((HOME/'JOB_MANIFEST.json').read_text())
    jobs = manifest['jobs']
    require([j['id'] for j in jobs] == ['c07_smoke_sft', 'c07_smoke_mace'], 'Exact two-job package required')
    require([j['hard_gpu_hours'] for j in jobs] == [0.5, 0.75], 'Reviewed caps changed')
    require(all(j['gpus'] == 1 and j['cpu_cores'] == 6 for j in jobs), 'Reviewed allocations changed')
    budget = json.loads((HOME/'BUDGET.json').read_text())
    require(budget['authorized_card_hours'] == 10 and budget['actual_card_hours'] == 0, 'Reconcile smoke spending first')
    require(budget['actual_plus_remaining_caps'] == 1.25, 'Unexpected reserved budget')
    require(not (HOME/'SUBMISSIONS.json').exists(), 'Existing ledger: reconcile only, never resubmit')
    hashes = {}
    for line in (HOME/'REVIEWED_INPUTS.sha256').read_text().splitlines():
        digest, name = line.split(None, 1)
        name = name.strip()
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe hash path')
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, 'Changed input: '+name)
        hashes[name] = digest
    dft = json.loads((ROOT/'calibration_gpu/claude07/SUBMISSIONS.json').read_text())
    require(len(dft['jobs']) == 8 and 'pending_submission' not in dft, 'Unreconciled DFT ledger')
    by_id = {str(j['slurm_job_id']): j for j in dft['jobs']}
    parents = {sid: set(map(str, j.get('dependencies_afterany', []) + j.get('dependencies_afterok', [])))
               for sid, j in by_id.items()}
    require(all(p <= set(by_id) for p in parents.values()), 'External DFT dependency needs review')
    tails = sorted(set(by_id) - set().union(*parents.values()), key=int)
    require(len(tails) == 2, 'Expected exactly two DFT resource lanes')
    covered = set()
    def visit(sid, stack):
        require(sid not in stack, 'Cyclic DFT dependency')
        covered.add(sid)
        for ancestor in parents[sid]:
            visit(ancestor, stack | {sid})
    for tail in tails:
        visit(tail, set())
    require(covered == set(by_id), 'DFT dependency coverage incomplete')
    return manifest, budget, hashes, dft, tails


def remote_code(manifest, budget, hashes, dft, tails, nonce):
    payload = dict(manifest=manifest, budget=budget, hashes=hashes, dft=dft,
                   tails=tails, nonce=nonce, dft_remote=DFT_REMOTE)
    return 'P='+repr(payload)+'\n'+r'''import datetime,hashlib,json,os,subprocess
from pathlib import Path
root=Path(P['manifest']['remote_smoke_root']);project=root.parent
ledger_path=root/'SUBMISSIONS.json'
def require(condition,message):
    if not condition:raise RuntimeError(message)
def persist(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)
require(not ledger_path.exists(),'Remote smoke ledger already exists; reconcile only')
require(not (root/'STOP_SMOKE.json').exists(),'Smoke pool stop remains active')
remote_dft=json.loads((Path(P['dft_remote'])/'SUBMISSIONS.json').read_text())
require(remote_dft==P['dft'],'Remote DFT ledger changed; recheck priority coverage')
require(not (Path(P['dft_remote'])/'CLAUDE07_POOL_STOP.json').exists(),'Reconcile DFT pool anomaly before adding work')
for name,digest in P['hashes'].items():
    require(hashlib.sha256((project/name).read_bytes()).hexdigest()==digest,'Remote input hash changed: '+name)
q=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j|%T']).decode()
require(not any('ysz_c07_smoke_' in line for line in q.splitlines()),'Smoke name already queued')
fd=os.open(str(root/'SUBMISSION_CLAIM.json'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as f:
    json.dump({'claim_id':P['nonce'],'approval_reference':'REVIEW_CLAUDE_07_gates.md'},f);f.flush();os.fsync(f.fileno())
budget_path=root/'BUDGET.json'
if budget_path.exists():
    require(json.loads(budget_path.read_text())==P['budget'],'Remote smoke budget changed')
else:
    fd=os.open(str(budget_path),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(P['budget'],f,indent=2);f.flush();os.fsync(f.fileno())
ledger={'approval_reference':'REVIEW_CLAUDE_07_gates.md','claim_id':P['nonce'],
 'pool':'TRAINING_SMOKE','no_automatic_retries':True,'DFT_priority':True,
 'covered_DFT_job_ids':[str(j['slurm_job_id']) for j in P['dft']['jobs']],
 'DFT_lane_tails':P['tails'],'jobs':[]}
persist(ledger_path,ledger)
for job in P['manifest']['jobs']:
    require(not (root/'STOP_SMOKE.json').exists(),'Smoke pool stop active')
    require('pending_submission' not in ledger,'Unreconciled submission intent')
    sub=root/job['sub'];directory=sub.parent
    require(not (directory/'submitted_job_id.txt').exists(),'Existing task ID')
    dep='afterany:'+':'.join(P['tails'])
    args=['--dependency='+dep,'--kill-on-invalid-dep=yes',str(sub)]
    test=subprocess.run(['sbatch','--test-only']+args,cwd=str(directory),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    print('TEST '+job['id']+' '+test.stdout.decode()+' '+test.stderr.decode(),flush=True)
    require(test.returncode==0,'Scheduler preflight rejected; no automatic retry')
    ledger['pending_submission']={'id':job['id'],'dependency':dep,
      'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};persist(ledger_path,ledger)
    result=subprocess.run(['sbatch','--parsable']+args,cwd=str(directory),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    require(result.returncode==0,'Submission rejected or ambiguous: '+result.stderr.decode())
    sid=result.stdout.decode().strip().split(';')[0]
    require(sid.isdigit(),'Ambiguous submission response; reconcile intent')
    require(not any(str(j['slurm_job_id'])==sid for j in ledger['jobs']),'Duplicate Slurm ID')
    (directory/'submitted_job_id.txt').write_text(sid+'\n')
    ledger['jobs'].append(dict(job,slurm_job_id=sid,dependency=dep,
      remote_output=str(root/'outputs'/(( 'sft_' if job['id'].endswith('_sft') else 'mace_')+sid)),
      submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    ledger.pop('pending_submission');persist(ledger_path,ledger)
    print('SUBMITTED '+job['id']+' '+sid,flush=True)
'''


def submit():
    manifest, budget, hashes, dft, tails = checked_inputs()
    nonce = uuid.uuid4().hex
    code = remote_code(manifest, budget, hashes, dft, tails, nonce)
    evidence = HOME/'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence/'exact_submission.py').write_text(code)
    try:
        result = ssh('your-cluster', 'python3 -', code.encode(), timeout=150)
        (evidence/'submission.txt').write_bytes(result.stdout+result.stderr)
        print(result.stdout.decode(), flush=True)
    finally:
        # Read-only reconciliation after success OR ambiguous remote interruption.
        result = ssh('your-cluster', 'cat '+shlex.quote(manifest['remote_smoke_root']+'/SUBMISSIONS.json'))
        ledger = json.loads(result.stdout)
        atomic_json(HOME/'SUBMISSIONS.json', ledger)
        require(ledger.get('claim_id') == nonce, 'Recovered ledger belongs to another runner; no retry')
        for row in ledger['jobs']:
            (HOME/row['sub']).parent.joinpath('submitted_job_id.txt').write_text(str(row['slurm_job_id'])+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if not args.submit:
        parser.error('--submit required; this is a one-shot approved operation')
    submit()
