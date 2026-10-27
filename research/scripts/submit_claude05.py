"""Explicit one-shot submissions for Claude05; GPU08 is gated on completed GPU07."""
import datetime,hashlib,io,json,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'calibration';G=ROOT/'calibration_gpu'
REF='REVIEW_CLAUDE_05_fast_gpu.md'

def ssh(host,command,payload=b'',timeout=60):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
                      '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',host,command],
                     input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    if r.returncode:raise RuntimeError(r.stdout.decode('utf8','replace')+'\n'+r.stderr.decode('utf8','replace'))
    return r

def submit_one(job,host,localdir,ledger_path,remote_root,is_gpu):
    for filename,digest in job['input_hashes'].items():
        assert hashlib.sha256((localdir/filename).read_bytes()).hexdigest()==digest
    if ledger_path.exists():
        localledger=json.loads(ledger_path.read_text())
        assert not any(j['name']==job['name'] for j in localledger['jobs']), 'Existing attempt: reconcile without retry'
    buf=io.BytesIO();hashlines=[]
    with tarfile.open(fileobj=buf,mode='w') as tar:
        for f in localdir.iterdir():
            assert f.is_file() and f.name not in ['POTCAR','submitted_job_id.txt']
            tar.add(f,arcname=job['name']+'/'+f.name,recursive=False)
            hashlines.append(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {job["name"]}/{f.name}')
        blob=('\n'.join(hashlines)+'\n').encode();i=tarfile.TarInfo(job['name']+'_UPLOAD.sha256');i.size=len(blob);tar.addfile(i,io.BytesIO(blob))
    r=ssh(host,'set -eu; cd '+remote_root+'; test ! -e '+job['name']+'; tar -xf -; sha256sum -c '+job['name']+'_UPLOAD.sha256',buf.getvalue())
    evidence=(G if is_gpu else C)/'evidence'
    (evidence/(job['name']+'_upload.txt')).write_bytes(r.stdout+r.stderr)
    code=r'''import datetime,json,subprocess
from pathlib import Path
root=Path(REMOTE_ROOT);name=JOB_NAME;d=root/name;ledger_path=root/LEDGER_NAME
is_gpu=IS_GPU
if ledger_path.exists():ledger=json.loads(ledger_path.read_text())
else:ledger={'approval_reference':'REVIEW_CLAUDE_05_fast_gpu.md','phase':'CLAUDE05_reference','hard_core_hours':192,'jobs':[]}
assert not any(j['name']==name for j in ledger['jobs'])
assert 'pending_submission' not in ledger, 'Uncertain previous attempt: reconcile first'
assert not (d/'submitted_job_id.txt').exists()
q=subprocess.check_output(['squeue','-u',str(__import__('os').environ['USER']),'-h','-o','%i|%j|%T']).decode('utf8','replace')
assert not any(name in row for row in q.splitlines()), 'Name already queued: do not resubmit'
r=subprocess.run(['sbatch','--test-only',str(d/'sub.sh')],cwd=d,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
print('TEST_ONLY '+r.stdout.decode('utf8','replace')+' '+r.stderr.decode('utf8','replace'),flush=True)
if r.returncode:raise RuntimeError('Scheduler preflight failed; no sbatch retry')
meta=json.loads((d/'job.json').read_text())
def persist():
    t=ledger_path.with_suffix('.json.tmp');t.write_text(json.dumps(ledger,indent=2)+'\n');t.replace(ledger_path)
ledger['active_approval_reference']='REVIEW_CLAUDE_05_fast_gpu.md';ledger['pending_submission']=name;persist()
r=subprocess.run(['sbatch','--parsable',str(d/'sub.sh')],cwd=d,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if r.returncode:raise RuntimeError('Submission rejected; inspect ledger, no automatic retry: '+r.stderr.decode('utf8','replace'))
jid=r.stdout.decode('utf8','replace').strip().split(';')[0];assert jid.isdigit()
(d/'submitted_job_id.txt').write_text(jid+'\n')
row={'name':name,'slurm_job_id':jid,'approval_reference':'REVIEW_CLAUDE_05_fast_gpu.md',
     'submitted_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'config':meta['config']}
if is_gpu:row.update(gpus=meta['gpus'],mpi_ranks=meta['mpi_ranks'],host_cpus=meta['host_cpus'],limit_hours=59/60,hard_card_hours=meta['hard_card_hours'],phase='CLAUDE05_gpu_recovery')
else:row.update(logical_job='ref_L00',phase='CLAUDE05_reference',cores=128,limit_hours=1.5,hard_core_hours=192)
ledger['jobs'].append(row);ledger.pop('pending_submission');persist()
print('SUBMITTED '+name+' '+jid,flush=True)
'''
    code=code.replace('REMOTE_ROOT',repr(remote_root)).replace('JOB_NAME',repr(job['name'])).replace('LEDGER_NAME',repr(ledger_path.name)).replace('IS_GPU',repr(is_gpu))
    (evidence/(job['name']+'_exact_submission.py')).write_text(code)
    try:
        r=ssh(host,'python3 -',code.encode(),timeout=90)
        (evidence/(job['name']+'_submission.txt')).write_bytes(r.stdout+r.stderr)
        print(r.stdout.decode('utf8','replace'),flush=True)
    finally:
        r=ssh(host,'if test -e '+remote_root+'/'+ledger_path.name+'; then cat '+remote_root+'/'+ledger_path.name+'; fi')
        if not r.stdout.strip():raise RuntimeError('No ledger created: preflight failed; do not resubmit without reconciliation')
        ledger=json.loads(r.stdout);ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
        for row in ledger['jobs']:
            if row['name']==job['name']:(localdir/'submitted_job_id.txt').write_text(row['slurm_job_id']+'\n')

def main():
    status=json.loads((C/'REVIEW_STATUS.json').read_text());assert status['approval_reference']==REF
    cpu=json.loads((C/'CLAUDE05_JOB_LIST.json').read_text())[0]
    gpu=json.loads((G/'CLAUDE05_JOB_LIST.json').read_text())
    if '--initial' in sys.argv:
        b=json.loads((C/'BUDGET.json').read_text());assert b['actual_plus_remaining_caps']<=3000
        assert status.get('cpu_ref_allowed') and status.get('gpu_07r_allowed')
        submit_one(cpu,'hzw-cpu',C/'execution_claude05'/cpu['name'],C/'REFERENCE_SUBMISSIONS.json',
                   '/online1/ycsc_hzw/hd100371/Jony/runs/01_calib',False)
        submit_one(gpu[0],'your-cluster',G/'execution_claude05'/gpu[0]['name'],G/'GPU_SUBMISSIONS.json',
                   '/CLUSTER_HOME/runs/01_calib_gpu',True)
    elif '--gpu08' in sys.argv:
        a=json.loads((G/'ANALYSIS.json').read_text());b=json.loads((G/'GPU_BUDGET.json').read_text())
        age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(a['analyzed_at_utc'])).total_seconds()
        assert 0<=age<300,'Refresh GPU evidence before gate decision'
        first=next(j for j in a['jobs'] if j['name']==gpu[0]['name'])
        assert first['accounting']['State']=='COMPLETED' and first['accounting']['ExitCode']=='0:0'
        assert first.get('vasp',{}).get('SCF_converged'),'07r SCF not independently checked'
        assert first['allocated_gpus']==1
        assert first.get('cuda_preflight',{}).get('passed') and first['cuda_preflight'].get('visible_GPU_count')==1
        assert not first['runtime_diagnostics']['cuInit_error_34'] and not first['runtime_diagnostics']['BRMIX_serious_count']
        assert b['actual_card_hours']+gpu[1]['hard_card_hours']<=2.9511
        submit_one(gpu[1],'your-cluster',G/'execution_claude05'/gpu[1]['name'],G/'GPU_SUBMISSIONS.json',
                   '/CLUSTER_HOME/runs/01_calib_gpu',True)
    else:raise SystemExit('Use --initial for exact09+07r, or --gpu08 after independent07rSCF gate')
    status=json.loads((C/'REVIEW_STATUS.json').read_text())
    refs=json.loads((C/'REFERENCE_SUBMISSIONS.json').read_text())
    ledger=json.loads((G/'GPU_SUBMISSIONS.json').read_text())
    status.update(reference_job_ids=[j['slurm_job_id'] for j in refs['jobs']],
                  gpu_job_ids=[j['slurm_job_id'] for j in ledger['jobs']],gpu_jobs_submitted=len(ledger['jobs']),
                  cpu_jobs_submitted=14+len(refs['jobs']),new_cloud_jobs_submitted=14+len(refs['jobs'])+len(ledger['jobs']),
                  p0_state='CLAUDE05_REFERENCE_AND_GPU_RUNNING',gpu_state='CLAUDE05_RECOVERY_RUNNING',
                  cpu_ref_allowed=False,gpu_07r_allowed=False,gpu_08r_submission_rule='Only after07r completed and checkedSCF; no other retries',
                  awaiting='Reference09, GPU07r then conditional08r results; P1/P2 held')
    (C/'REVIEW_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')

if __name__=='__main__':main()
