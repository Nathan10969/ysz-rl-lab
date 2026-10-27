"""Read and reconcile only the two GPU pilot attempts; never submits or retries."""
import csv,datetime,io,json,re,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];G=ROOT/'calibration_gpu'
ledger=json.loads((G/'GPU_SUBMISSIONS.json').read_text())
code=r'''import datetime,json,subprocess
from pathlib import Path
root=Path('/CLUSTER_HOME/runs/01_calib_gpu')
ledger=json.loads((root/'GPU_SUBMISSIONS.json').read_text())
def cmd(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=25)
 return {'code':p.returncode,'stdout':p.stdout.decode('utf8','replace'),'stderr':p.stderr.decode('utf8','replace')}
ids=','.join(j['slurm_job_id'] for j in ledger['jobs'])
out={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'ledger':ledger,'sacct':cmd(['sacct','-j',ids,'-P','--units=K','-o','JobIDRaw,JobName%40,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES%170,MaxRSS,Start,End']),
 'squeue':cmd(['squeue','-j',ids,'-h','-o','%i|%T|%M|%C|%b|%R']),'files':{}}
for j in ledger['jobs']:
 d=root/j['name'];assert (d/'submitted_job_id.txt').read_text().strip()==j['slurm_job_id']
 data={}
 for name in ['OUTCAR','OSZICAR','vasprun.xml','CONTCAR','runtime_input_hashes.sha256','log.txt','gpu_samples.csv','cuda_preflight_filtered.json','cuda_preflight_preload.json','submitted_job_id.txt','slurm-'+j['slurm_job_id']+'.out']:
  p=d/name
  if p.is_file():
   assert p.stat().st_size<50000000
   data[name]=p.read_bytes().decode('utf8','replace')
 out['files'][j['name']]=data
print(json.dumps(out))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
                  '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','your-cluster','python3 -'],
                 input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=75)
if r.returncode:raise RuntimeError(r.stderr.decode('utf8','replace'))
snapshot=json.loads(r.stdout);assert snapshot['ledger']==ledger
assert snapshot['sacct']['code']==0,snapshot['sacct']
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
meta={k:v for k,v in snapshot.items() if k!='files'}
(G/'LATEST.json').write_text(json.dumps(meta,indent=2)+'\n')
(G/'evidence'/f'snapshot_{stamp}.json').write_text(json.dumps(meta,indent=2)+'\n')
lines=snapshot['sacct']['stdout'].splitlines();head=lines[0].split('|')
accounts=[dict(zip(head,line.split('|'))) for line in lines[1:]]
parents={row['JobIDRaw']:row for row in accounts if row['JobIDRaw'] in {j['slurm_job_id'] for j in ledger['jobs']}}
terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
results=[];actual=0;remaining=0
cpu_analysis=json.loads((ROOT/'calibration/P0_ANALYSIS.json').read_text())
cpu=next(j for j in cpu_analysis['attempts'] if j['slurm_job_id']=='1882636')
for j in ledger['jobs']:
 data=snapshot['files'][j['name']];d=G/'outputs'/j['name'];d.mkdir(parents=True,exist_ok=True)
 for name,text in data.items():(d/name).write_text(text)
 acc=parents.get(j['slurm_job_id'],{})
 state=acc.get('State','NOT_YET_IN_ACCOUNTING').split()[0]
 tres=dict(item.split('=',1) for item in acc.get('AllocTRES','').split(',') if '=' in item)
 allocated_gpus=int(tres.get('gres/gpu',tres.get('gres/gpu:a100',0)))
 seconds=int(acc.get('ElapsedRaw') or 0)
 assert allocated_gpus in [0,j['gpus']],acc
 if seconds:assert allocated_gpus==j['gpus'],'Unknown GPU allocation: do not assume zero cost'
 cost=allocated_gpus*seconds/3600;actual+=cost
 if state not in terminal:remaining+=max(0,j['hard_card_hours']-cost)
 result=dict(j,accounting=acc,allocated_gpus=allocated_gpus,actual_card_hours=cost,
             step_accounting=[a for a in accounts if a['JobIDRaw'].startswith(j['slurm_job_id']+'.')])
 for filename in ['cuda_preflight_filtered.json','cuda_preflight_preload.json']:
  if filename in data:
   result['cuda_preflight']=json.loads(data[filename]);result['cuda_preflight']['source']=filename
 log=data.get('log.txt','');out=data.get('OUTCAR','')
 result['runtime_diagnostics']={'cuInit_error_34':'cuInit returned error 34' in log,
  'MPI_insufficient_slots':'not enough slots available' in log,
  'BRMIX_serious_count':log.count('BRMIX: very serious problems'),
  'MPI_thread_header':re.findall(r'running\s+\d+\s+mpi-ranks[^\n]*',log),
  'GPU_initialization_lines':[l for l in log.splitlines() if any(k in l.lower() for k in ['offloading','gpus detected','accelerator fatal'])]}
 times=[float(x) for x in re.findall(r'LOOP:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)',out)]
 result['SCF_loop_real_seconds']=times
 result['SCF_loop_real_seconds_mean']=float(np.mean(times)) if times else None
 if 'gpu_samples.csv' in data:
  samples=list(csv.DictReader(io.StringIO(data['gpu_samples.csv']),skipinitialspace=True))
  # The sampler uses csv with units; values such as "10111 MiB" are valid observations.
  mem=[float(match.group(1)) for row in samples
       for match in [re.fullmatch(r'\s*([\d.]+)(?:\s+MiB)?\s*',row.get('memory.used [MiB]',''))] if match]
  result['sampled_max_GPU_memory_MiB']=max(mem) if mem else None
 if state=='COMPLETED' and acc.get('ExitCode')=='0:0' and 'vasprun.xml' in data:
  xml=ET.fromstring(data['vasprun.xml']);calcs=xml.findall('calculation');calc=calcs[-1]
  forces=np.array([[float(v) for v in row.text.split()] for row in calc.findall('varray[@name="forces"]/v')])
  positions=np.array([[float(v) for v in row.text.split()] for row in calc.findall('structure/varray[@name="positions"]/v')])
  e={row.attrib['name']:float(row.text) for row in calc.findall('energy/i')}
  nelm=int(xml.find('.//parameters//i[@name="NELM"]').text)
  good='aborting loop because EDIFF is reached' in out and len(calc.findall('scstep'))<nelm
  result['vasp']={'SCF_converged':good,'n_scf_steps':len(calc.findall('scstep')),'energy_eV':e,
    'forces_eV_A':forces.tolist(),'fractional_positions':positions.tolist(),
    'max_force_eV_A':float(np.linalg.norm(forces,axis=1).max())}
  if good and cpu.get('vasp',{}).get('SCF_converged'):
   assert np.allclose(positions,cpu['vasp']['fractional_positions'],atol=1e-8)
   diff=forces-np.array(cpu['vasp']['forces_eV_A'])
   de=e['e_0_energy']-cpu['vasp']['energy_eV']['e_0_energy']
   result['comparison_to_CPU_06f']={'delta_E0_eV':de,'absolute_energy_difference_at_most_0p01_eV_cell':abs(de)<=.01,
    'max_atom_force_difference_eV_A':float(np.linalg.norm(diff,axis=1).max()),
    'force_component_RMSE_eV_A':float(np.sqrt(np.mean(diff**2))),
    'wall_speedup_over_CPU':float(cpu['accounting']['ElapsedRaw'])/seconds}
   result['comparison_to_CPU_06f']['max_force_difference_at_most_0p02_eV_A']=float(np.linalg.norm(diff,axis=1).max())<=.02
   result['comparison_to_CPU_06f']['passed']=abs(de)<=.01 and float(np.linalg.norm(diff,axis=1).max())<=.02
 results.append(result)
budget=json.loads((G/'GPU_BUDGET.json').read_text())
if budget.get('recovery_authorized') and (G/'CLAUDE05_JOB_LIST.json').exists():
 recovery=json.loads((G/'CLAUDE05_JOB_LIST.json').read_text())
 first=next((r for r in results if r['name']==recovery[0]['name']),None)
 first_failed=bool(first and first['accounting'].get('State','').split()[0] in terminal and
                   not (first['accounting'].get('State')=='COMPLETED' and first.get('vasp',{}).get('SCF_converged')))
 submitted={r['name'] for r in results}
 for job in recovery:
  if job['name'] not in submitted and not first_failed:remaining+=job['hard_card_hours']
 budget['conditional_08r_blocked_by_07r_failure']=first_failed
budget.update(actual_card_hours=actual,remaining_card_hour_caps=remaining,actual_plus_remaining_caps=actual+remaining,
 unspent_authorized_card_hours=budget['authorized_card_hours']-actual,updated_at_utc=snapshot['checked_at_utc'])
(G/'GPU_BUDGET.json').write_text(json.dumps(budget,indent=2)+'\n')
assert actual+remaining<=3,'GPU pilot budget risk: no further jobs'
analysis={'analyzed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':results,
 'all_terminal':all(j['accounting'].get('State','').split()[0] in terminal for j in results),
 'all_SCF_converged':all(j.get('vasp',{}).get('SCF_converged') for j in results)}
(G/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n')
with (G/'SCF_TIMINGS.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['job','slurm_job_id','SCF_step_1based','wall_seconds_rank0'])
 for j in results:
  for n,t in enumerate(j['SCF_loop_real_seconds'],1):w.writerow([j['name'],j['slurm_job_id'],n,t])
print(json.dumps({'jobs':[{k:j.get(k) for k in ['name','slurm_job_id','accounting','actual_card_hours','runtime_diagnostics','comparison_to_CPU_06f']} for j in results],
 'GPU_card_hours':actual,'all_terminal':analysis['all_terminal']},indent=2))
