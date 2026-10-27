"""Read-only SSH polling for the five authorized P0 jobs; never submits/restarts."""
import datetime,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'calibration'
ledger=json.loads((C/'P0_SUBMISSIONS.json').read_text())
p1ledger=json.loads((C/'P1_SUBMISSIONS.json').read_text()) if (C/'P1_SUBMISSIONS.json').exists() else {'jobs':[]}
fastledger=json.loads((C/'FAST_SUBMISSIONS.json').read_text()) if (C/'FAST_SUBMISSIONS.json').exists() else {'jobs':[]}
refledger=json.loads((C/'REFERENCE_SUBMISSIONS.json').read_text()) if (C/'REFERENCE_SUBMISSIONS.json').exists() else {'jobs':[]}
all_attempts=ledger['jobs']+p1ledger['jobs']+fastledger['jobs']+refledger['jobs']
ids=[j['slurm_job_id'] for j in all_attempts]
remote=r'''
import json,subprocess,datetime
from pathlib import Path
root=Path('/online1/ycsc_hzw/hd100371/Jony/runs/01_calib')
ledger=json.loads((root/'P0_SUBMISSIONS.json').read_text())
if (root/'P1_SUBMISSIONS.json').exists():
    ledger['jobs']+=json.loads((root/'P1_SUBMISSIONS.json').read_text())['jobs']
if (root/'FAST_SUBMISSIONS.json').exists():
    ledger['jobs']+=json.loads((root/'FAST_SUBMISSIONS.json').read_text())['jobs']
if (root/'REFERENCE_SUBMISSIONS.json').exists():
    ledger['jobs']+=json.loads((root/'REFERENCE_SUBMISSIONS.json').read_text())['jobs']
def command(args):
    p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        out,err=p.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        p.kill();out,err=p.communicate()
        return dict(code=124,stdout=out.decode('utf-8','replace'),stderr=err.decode('utf-8','replace')+'; read-only query timed out after 20 seconds')
    return dict(code=p.returncode,stdout=out.decode('utf-8','replace'),stderr=err.decode('utf-8','replace'))
ids=','.join(j['slurm_job_id'] for j in ledger['jobs'])
account=command(['sacct','-j',ids,'-P','--units=K','-o','JobIDRaw,JobName%45,State,ExitCode,ElapsedRaw,AllocCPUS,CPUTimeRAW,TotalCPU,MaxRSS,AveRSS,MaxVMSize,NodeList,Start,End'])
queue=command(['squeue','-j',ids,'-h','-o','%i|%T|%M|%C|%R'])
active=','.join(line.split('|')[0] for line in queue['stdout'].splitlines() if '|RUNNING|' in line)
result={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'sacct':account,'squeue':queue,
 'sstat':command(['sstat','-a','-j',active,'-P','-o','JobID,NTasks,MaxRSS,AveRSS,MaxVMSize']) if active else {'code':0,'stdout':'','stderr':''},
 'files':{}}
for j in ledger['jobs']:
    d=root/j['name'];data={}
    for name in ['OSZICAR','OUTCAR','slurm-'+j['slurm_job_id']+'.out']:
        p=d/name
        if p.exists():
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-5000));data[name]=f.read().decode('utf-8','replace')
    result['files'][j['name']]=data
print(json.dumps(result))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
                  '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',
                  'hzw-cpu','python3 -'],input=remote.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=75)
if r.returncode:
    raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
data=json.loads(r.stdout)
data['ssh_stderr']=r.stderr.decode('utf8',errors='replace')
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
(C/'evidence'/f'P0_snapshot_{stamp}.json').write_text(json.dumps(data,indent=2)+'\n')
if data['sacct']['code']!=0:raise RuntimeError(data['sacct'])
(C/'P0_LATEST.json').write_text(json.dumps(data,indent=2)+'\n')
lines=data['sacct']['stdout'].splitlines();head=lines[0].split('|');rows=[dict(zip(head,l.split('|'))) for l in lines[1:]]
parents={r['JobIDRaw']:r for r in rows if r['JobIDRaw'] in ids}
jobs=json.loads((C/'jobs.json').read_text());budget=json.loads((C/'BUDGET.json').read_text())
finished={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
actual=sum(float(r['CPUTimeRAW'] or 0)/3600 for r in parents.values())
p0ids={j['slurm_job_id'] for j in ledger['jobs']}
p0actual=sum(float(r['CPUTimeRAW'] or 0)/3600 for jid,r in parents.items() if jid in p0ids)
p1ids={j['slurm_job_id'] for j in p1ledger['jobs']}
p1actual=sum(float(r['CPUTimeRAW'] or 0)/3600 for jid,r in parents.items() if jid in p1ids)
fastids={j['slurm_job_id'] for j in fastledger['jobs']}
fastactual=sum(float(r['CPUTimeRAW'] or 0)/3600 for jid,r in parents.items() if jid in fastids)
refids={j['slurm_job_id'] for j in refledger['jobs']}
refactual=sum(float(r['CPUTimeRAW'] or 0)/3600 for jid,r in parents.items() if jid in refids)
logical_caps=ledger.get('p0_remaining_logical_caps',{j['id']:j['hard_core_hours'] for j in jobs if j['phase']=='P0_cost'})
remaining=0
for logical,cap in logical_caps.items():
    attempts=[s for s in ledger['jobs'] if s.get('logical_job',s['name'])==logical]
    last=attempts[-1]
    row=parents.get(last['slurm_job_id'],{})
    if row.get('State')=='COMPLETED':continue
    # Reserve a future replacement for a failed/cancelled point too; never submit it here.
    running=row.get('State','').split()[0] not in finished if row else False
    remaining+=max(0,cap-(float(row.get('CPUTimeRAW') or 0)/3600 if running else 0))
claude03=json.loads((C/'CLAUDE03_JOB_LIST.json').read_text()) if (C/'CLAUDE03_JOB_LIST.json').exists() else []
p1caps={j['logical_job']:j['hard_core_hours'] for j in claude03 if j['phase']=='P1_small'}
p1remaining=0
for logical,cap in p1caps.items():
    attempts=[s for s in p1ledger['jobs'] if s.get('logical_job',s['name'])==logical]
    row=parents.get(attempts[-1]['slurm_job_id'],{}) if attempts else {}
    if row.get('State')=='COMPLETED':continue
    active=row.get('State','').split()[0] not in finished if row else False
    p1remaining+=max(0,cap-(float(row.get('CPUTimeRAW') or 0)/3600 if active else 0))
if budget.get('cpu_future_scope')=='94_ATOM_ONLY_AFTER_GPU_SELECTION':
    later_fixed=sum(j['hard_core_hours'] for j in jobs if j['n_atoms']==94 and j['phase'] not in ['P0_cost','P1_small'])
else:
    later_fixed=(2496 if p1caps else 2672)-budget.get('fast_reallocated_from_unreleased_p2plus_core_hours',0)
later_remaining=later_fixed+p1remaining
fastjobs=json.loads((C/'CLAUDE04_JOB_LIST.json').read_text()) if (C/'CLAUDE04_JOB_LIST.json').exists() else []
fastremaining=0
for job in fastjobs:
    attempts=[s for s in fastledger['jobs'] if s['name']==job['name']]
    row=parents.get(attempts[-1]['slurm_job_id'],{}) if attempts else {}
    if row.get('State','').split()[0:1] and row['State'].split()[0] in finished:continue
    fastremaining+=max(0,job['hard_core_hours']-float(row.get('CPUTimeRAW') or 0)/3600)
refjobs=json.loads((C/'CLAUDE05_JOB_LIST.json').read_text()) if (C/'CLAUDE05_JOB_LIST.json').exists() else []
refremaining=0
for job in refjobs:
    attempts=[s for s in refledger['jobs'] if s['name']==job['name']]
    row=parents.get(attempts[-1]['slurm_job_id'],{}) if attempts else {}
    if row.get('State','').split()[0:1] and row['State'].split()[0] in finished:continue
    refremaining+=max(0,job['hard_core_hours']-float(row.get('CPUTimeRAW') or 0)/3600)
budget.update(actual_new_job_core_hours=actual,actual_p0_core_hours=p0actual,actual_p1_core_hours=p1actual,
              fast_actual_core_hours=fastactual,remaining_fast_core_hour_caps=fastremaining,
              reference_actual_core_hours=refactual,remaining_reference_core_hour_caps=refremaining,
              cumulative_actual_core_hours=actual+budget['smoke_used_core_hours'],
              p0_status={s['name']:parents.get(s['slurm_job_id'],{'State':'NOT_YET_IN_ACCOUNTING'}) for s in ledger['jobs']},
              p1_status={s['name']:parents.get(s['slurm_job_id'],{'State':'NOT_YET_IN_ACCOUNTING'}) for s in p1ledger['jobs']},
              fast_status={s['name']:parents.get(s['slurm_job_id'],{'State':'NOT_YET_IN_ACCOUNTING'}) for s in fastledger['jobs']},
              reference_status={s['name']:parents.get(s['slurm_job_id'],{'State':'NOT_YET_IN_ACCOUNTING'}) for s in refledger['jobs']},
              unfinished_p0_remaining_core_hour_caps=remaining,
              unsubmitted_later_phases_hard_core_hours=later_remaining,
              remaining_p1_core_hour_caps=p1remaining,remaining_p2plus_core_hour_caps=later_fixed,
              actual_plus_remaining_caps=budget['smoke_used_core_hours']+actual+remaining+later_remaining+fastremaining+refremaining,
              p0_actual_plus_remaining_caps=p0actual+remaining,
              p1_actual_plus_remaining_caps=p1actual+p1remaining,
              fast_actual_plus_remaining_caps=fastactual+fastremaining,
              reference_actual_plus_remaining_caps=refactual+refremaining,
              accounting_updated_at_utc=data['checked_at_utc'])
budget['reserve_after_current_plan_caps']=budget['authorized_core_hours']-budget['actual_plus_remaining_caps']
(C/'BUDGET.json').write_text(json.dumps(budget,indent=2)+'\n')
assert budget['actual_plus_remaining_caps']<=3000
assert budget['p0_actual_plus_remaining_caps']<=216
print(data['squeue']['stdout'].strip())
print(data['sacct']['stdout'].strip())
print('Calibration used core-hours:',round(actual,6),'P0:',round(p0actual,6),'P1:',round(p1actual,6),'incl smoke:',round(budget['cumulative_actual_core_hours'],6))
for name,files in data['files'].items():
    osc=files.get('OSZICAR','').splitlines()
    if osc:print(name,osc[-1])
