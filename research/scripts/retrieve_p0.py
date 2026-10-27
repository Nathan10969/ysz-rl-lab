"""Copy observed outputs, including live progress; analysis accepts energies only after completion."""
import io,json,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'calibration'
snapshot=json.loads((C/'P0_LATEST.json').read_text())
rows=snapshot['sacct']['stdout'].splitlines();head=rows[0].split('|')
states={r['JobIDRaw']:r['State'] for r in [dict(zip(head,l.split('|'))) for l in rows[1:]]}
ledger=json.loads((C/'P0_SUBMISSIONS.json').read_text())
if (C/'P1_SUBMISSIONS.json').exists():ledger['jobs']+=json.loads((C/'P1_SUBMISSIONS.json').read_text())['jobs']
if (C/'FAST_SUBMISSIONS.json').exists():ledger['jobs']+=json.loads((C/'FAST_SUBMISSIONS.json').read_text())['jobs']
if (C/'REFERENCE_SUBMISSIONS.json').exists():ledger['jobs']+=json.loads((C/'REFERENCE_SUBMISSIONS.json').read_text())['jobs']
terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
targets=[j for j in ledger['jobs'] if states.get(j['slurm_job_id'],'PENDING').split()[0] in terminal|{'RUNNING','COMPLETING'}]
code="""import sys,tarfile,json
from pathlib import Path
root=Path('/online1/ycsc_hzw/hd100371/Jony/runs/01_calib')
targets=json.loads(sys.stdin.buffer.readline())
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for j in targets:
  for name in ['OUTCAR','OSZICAR','vasprun.xml','CONTCAR','runtime_input_hashes.sha256','submitted_job_id.txt','slurm-'+j['slurm_job_id']+'.out']:
   p=root/j['name']/name
   if p.is_file():tar.add(str(p),arcname=j['name']+'/'+name,recursive=False)
"""
import shlex
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','hzw-cpu',
                  'python3 -c '+shlex.quote(code)],input=(json.dumps(targets)+'\n').encode(),
                 stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
r.check_returncode()
with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tar:
 for member in tar.getmembers():
  path=Path(member.name)
  assert member.isfile() and not path.is_absolute() and '..' not in path.parts
  assert path.parts[0] in {j['name'] for j in targets}
  assert path.name!='POTCAR' and member.size<100_000_000
  dest=C/'P0_outputs'/path;dest.parent.mkdir(parents=True,exist_ok=True)
  dest.write_bytes(tar.extractfile(member).read())
print('Retrieved outputs (running attempts remain partial):',[j['name'] for j in targets])
