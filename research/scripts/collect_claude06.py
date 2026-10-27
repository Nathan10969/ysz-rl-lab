"""Collect Claude06 evidence, guard failures, and account separate compute pools."""
import argparse,csv,datetime,io,json,re,tarfile,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from ase.io import read
from submit_claude05 import ssh
from claude06_safety import TERMINAL,budget_projection,validate_output
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'calibration';G=ROOT/'calibration_gpu';REF='REVIEW_CLAUDE_06_release.md'
FILES=['OUTCAR','OSZICAR','CONTCAR','vasprun.xml','runtime_input_hashes.sha256','RESULT_CHECK.json',
       'gpu_samples.csv','cuda_preflight_filtered.json','cuda_preflight_preload.json','submitted_job_id.txt','log.txt']

def save(p,data):p.write_text(json.dumps(data,indent=2)+'\n')

def fetch_snapshot(home):
    gpu=home==G;host='your-cluster' if gpu else 'hzw-cpu'
    manifest=json.loads((home/'CLAUDE06_JOB_LIST.json').read_text())
    root=str(Path(manifest[0]['remote_directory']).parent)
    local=json.loads((home/'CLAUDE06_SUBMISSIONS.json').read_text()) if (home/'CLAUDE06_SUBMISSIONS.json').exists() else {'jobs':[]}
    code=r'''import datetime,json,os,subprocess
from pathlib import Path
root=Path(REMOTE_ROOT)
ledger=json.loads((root/'CLAUDE06_SUBMISSIONS.json').read_text()) if (root/'CLAUDE06_SUBMISSIONS.json').exists() else {'jobs':[]}
def run(args):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=25)
    return dict(code=p.returncode,stdout=p.stdout.decode('utf8','replace'),stderr=p.stderr.decode('utf8','replace'))
ids=','.join(j['slurm_job_id'] for j in ledger['jobs'])
stop=None
stop_path=root/'CLAUDE06_STOP.json'
if stop_path.exists():
    stop_raw=stop_path.read_text()
    try:stop=json.loads(stop_raw)
    except ValueError as error:
        # Keep a malformed stop marker active and preserve its exact evidence.
        # A result-recovery parser must never clear the submission stop.
        stop={'stop_marker_present':True,'parse_error':str(error),'raw_text':stop_raw}
result={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ledger':ledger,
 'account':run(['sacct','-j',ids,'-P','--units=K','-o','JobIDRaw,JobName%60,State,ExitCode,ElapsedRaw,AllocCPUS,NTasks,CPUTimeRAW,AllocTRES%150,MaxRSS,NodeList,Start,End']) if ids else {'code':0,'stdout':'','stderr':''},
 'queue':run(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j|%T|%C|%b|%R']),
 'stop':stop,'files':{}}
for j in ledger['jobs']:
    d=root/j['name'];result['files'][j['name']]={}
    for name in ['OSZICAR','OUTCAR','log.txt','slurm-'+j['slurm_job_id']+'.out']:
        p=d/name
        if p.exists():
            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-8000));tail=f.read().decode('utf8','replace')
            result['files'][j['name']][name]=dict(bytes=p.stat().st_size,tail=tail)
print(json.dumps(result))
'''.replace('REMOTE_ROOT',repr(root))
    r=ssh(host,'python3 -',code.encode(),timeout=100);data=json.loads(r.stdout)
    if data['account']['code'] or data['queue']['code']:
        save(home/'evidence/Claude06_query_error.json',data)
        raise RuntimeError(json.dumps({'account':data['account'],'queue':data['queue']}))
    # A remote-only ID is recovered locally, never submitted again.
    assert all(j in data['ledger']['jobs'] for j in local['jobs']),'Local/remote ledger conflict'
    if data['ledger']['jobs']:save(home/'CLAUDE06_SUBMISSIONS.json',data['ledger'])
    save(home/'CLAUDE06_LATEST.json',data)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    save(home/'evidence'/('Claude06_snapshot_'+stamp+'.json'),data)
    lines=data['account']['stdout'].splitlines()
    head=lines[0].split('|') if lines else [];rows=[dict(zip(head,line.split('|'))) for line in lines[1:]]
    ids={j['slurm_job_id'] for j in data['ledger']['jobs']}
    data['parents']={r['JobIDRaw']:r for r in rows if r['JobIDRaw'] in ids}
    data['steps']=[r for r in rows if '.' in r.get('JobIDRaw','')]
    return host,root,manifest,data

def retrieve(home,host,root,data):
    output=home/'claude06_outputs';output.mkdir(exist_ok=True);wanted=[]
    for job in data['ledger']['jobs']:
        account=data['parents'].get(job['slurm_job_id'],{});state=account.get('State','UNKNOWN').split()[0]
        if state not in TERMINAL:continue
        marker=output/job['name']/'RETRIEVED_TERMINAL.json'
        if marker.exists() and json.loads(marker.read_text()).get('state')==state:continue
        wanted.append(dict(job,files=FILES+['slurm-'+job['slurm_job_id']+'.out'],state=state))
    if not wanted:return
    code=r'''import sys,tarfile
from pathlib import Path
root=Path(REMOTE_ROOT)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
    for job in JOBS:
        for name in job['files']:
            f=root/job['name']/name
            if f.is_file():tar.add(str(f),arcname=job['name']+'/'+name,recursive=False)
'''.replace('REMOTE_ROOT',repr(root)).replace('JOBS',repr(wanted))
    r=ssh(host,'python3 -',code.encode(),timeout=180)
    with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:') as tar:
        for item in tar:
            parts=Path(item.name).parts
            assert len(parts)==2 and parts[0] in [j['name'] for j in wanted] and item.isfile()
            dest=output/item.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(tar.extractfile(item).read())
    for job in wanted:save(output/job['name']/'RETRIEVED_TERMINAL.json',dict(state=job['state'],at_utc=data['checked_at_utc']))

def analyze_one(home,job,attempt,data):
    account=data['parents'].get(attempt['slurm_job_id'],{});state=account.get('State','UNKNOWN').split()[0]
    elapsed=float(account.get('ElapsedRaw') or 0)
    # Only parent allocation time is billed; MPI steps are diagnostics, not extra charges.
    allocated_gpus=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',account.get('AllocTRES',''))
    ngpu=int(allocated_gpus.group(1)) if allocated_gpus else job['gpus']
    if elapsed>0 and state in ['RUNNING','COMPLETED']:
        assert int(account['AllocCPUS'])==job['cores'],'Unexpected CPU allocation: reconcile billing'
        if job['gpus']:assert allocated_gpus and ngpu==1,'GPU allocation must be independently verified as1'
    actual=ngpu*elapsed/3600 if job['gpus'] else float(account.get('CPUTimeRAW') or 0)/3600
    result=dict(job,slurm_job_id=attempt['slurm_job_id'],state=state,accounting=account,actual=actual,
                step_accounting=[r for r in data['steps'] if r['JobIDRaw'].startswith(attempt['slurm_job_id']+'.')],usable=False)
    logs=data['files'].get(job['name'],{});tails='\n'.join(f['tail'] for f in logs.values())
    result['tail_BRMIX_serious_count']=tails.count('BRMIX: very serious problems')
    scf=re.findall(r'^\s*(?:DAV|RMM|CG):\s*(\d+)\s+',logs.get('OSZICAR',{}).get('tail',''),re.M)
    if not scf:scf=re.findall(r'^\s*(?:DAV|RMM|CG):\s*(\d+)\s+',logs.get('log.txt',{}).get('tail',''),re.M)
    result['last_printed_SCF_step']=int(scf[-1]) if scf else None
    d=home/'claude06_outputs'/job['name']
    if state=='COMPLETED' and account.get('ExitCode')=='0:0' and (d/'vasprun.xml').exists():
        try:
            text=(d/'vasprun.xml').read_text(errors='replace');out=(d/'OUTCAR').read_text(errors='replace')
            result['convergence']=validate_output(text,out,job['kind']=='relax')
            result['usable']=result['convergence']['valid']
            xml=ET.fromstring(text);last=xml.findall('calculation')[-1]
            result['energy_eV']={i.attrib['name']:float(i.text) for i in last.findall('energy/i')}
            forces=np.array([[float(x) for x in v.text.split()] for v in last.findall('varray[@name="forces"]/v')])
            assert forces.shape==(job['n_atoms'],3) and np.all(np.isfinite(forces))
            assert all(np.isfinite(value) for value in result['energy_eV'].values())
            result['forces_eV_A']=forces.tolist();result['max_force_eV_A']=float(np.linalg.norm(forces,axis=1).max())
            result['SCF_loop_real_seconds']=[float(x) for x in re.findall(r'LOOP:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)',out)]
            if job['kind']=='sp':
                xyz=C/'claude06_pool'/f'{job["config"]}_relaxed.extxyz' if job['phase']=='P2b_maincell' else C/'geometries'/f'{job["config"]}_relaxed.extxyz'
                mace=read(xyz);symbols=[rc.find('c').text.strip() for rc in xml.findall('atominfo/array[@name="atoms"]/set/rc')]
                positions=np.array([[float(x) for x in v.text.split()] for v in last.findall('structure/varray[@name="positions"]/v')])
                delta=positions-mace.get_scaled_positions(wrap=False);delta-=np.rint(delta)
                assert symbols==mace.get_chemical_symbols() and np.max(np.linalg.norm(researcher@example.org,axis=1))<1e-6
                mf=mace.get_forces();diff=forces-mf
                result['MACE_energy_eV']=float(mace.get_potential_energy())
                result['MACE_force_component_RMSE_eV_A']=float(np.sqrt(np.mean(diff**2)))
                result['MACE_atom_force_errors_eV_A']=np.linalg.norm(diff,axis=1).tolist()
        except Exception as error:
            result.update(usable=False,parse_error=repr(error))
    return result

def guard(allsets):
    failed=[]
    for home,host,root,manifest,data,results in allsets:
        failed += [r['name'] for r in results if (r['state'] in TERMINAL and r['state']!='COMPLETED') or
                   (r['state']=='COMPLETED' and not r['usable'])]
        failed += [r['name']+': repeated seriousBRMIX requires review' for r in results if r['state']=='RUNNING' and r['tail_BRMIX_serious_count']>=6]
        if data['stop']:failed.append(str(data['stop']))
    if not failed:return []
    for home,host,root,manifest,data,results in allsets:
        code=r'''import json,subprocess
from pathlib import Path
root=Path(REMOTE_ROOT);stop=root/'CLAUDE06_STOP.json'
if not stop.exists():stop.write_text(json.dumps({'collector_detected_failures':FAILURES},indent=2)+'\n')
ids={j['slurm_job_id'] for j in JOBS}
q=subprocess.check_output(['squeue','-h','-o','%i|%T']).decode()
pending=[line.split('|')[0] for line in q.splitlines() if line.split('|')[0] in ids and line.split('|')[1]=='PENDING']
if pending:subprocess.check_call(['scancel']+pending)
print(json.dumps({'cancelled_only_pending_own_ids':pending}))
'''.replace('REMOTE_ROOT',repr(root)).replace('FAILURES',repr(failed)).replace('JOBS',repr(data['ledger']['jobs']))
        r=ssh(host,'python3 -',code.encode());save(home/'evidence/Claude06_guard_last.json',json.loads(r.stdout))
    return failed

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--retrieve',action='store_true');parser.add_argument('--guard',action='store_true');args=parser.parse_args()
    sets=[];observed={}
    for home in [C,G]:
        host,root,manifest,data=fetch_snapshot(home)
        if args.retrieve:retrieve(home,host,root,data)
        byname={j['name']:j for j in manifest}
        results=[analyze_one(home,byname[a['name']],a,data) for a in data['ledger']['jobs']]
        save(home/'CLAUDE06_ANALYSIS.json',dict(checked_at_utc=data['checked_at_utc'],jobs=results))
        with (home/'CLAUDE06_SCF_TIMINGS.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['name','slurm_job_id','SCF_step_global','wall_seconds_rank0'])
            for r in results:
                for i,seconds in enumerate(r.get('SCF_loop_real_seconds',[]),1):writer.writerow([r['name'],r['slurm_job_id'],i,seconds])
        observed[home]={r['name']:dict(state=r['state'],actual=r['actual']) for r in results}
        sets.append((home,host,root,manifest,data,results))
    failures=guard(sets) if args.guard else []
    for home,host,root,manifest,data,results in sets:
        isgpu=home==G;alljobs=list(manifest)
        if isgpu and len(manifest)==9:alljobs += [dict(name='reserved_P2b_'+str(i),cap=.25) for i in range(16)]
        projection=budget_projection(alljobs,observed[home],.07416666666666666 if isgpu else 253.68)
        assert projection['actual_plus_remaining_caps']<=(40 if isgpu else 3000),'Budget risk: stop expansion'
        path=home/('GPU_BUDGET.json' if isgpu else 'BUDGET.json');b=json.loads(path.read_text())
        b.update(approval_reference=REF,actual_plus_remaining_caps=projection['actual_plus_remaining_caps'],
                 accounting_updated_at_utc=data['checked_at_utc'])
        if isgpu:b.update(authorized_card_hours=40,actual_card_hours=projection['actual'],remaining_card_hour_caps=projection['remaining_caps'],
                 unspent_authorized_card_hours=40-projection['actual'],uncommitted_after_all_caps=40-projection['actual_plus_remaining_caps'])
        else:b.update(cumulative_actual_core_hours=projection['actual'],claude06_actual_core_hours=projection['actual']-253.68,
                 remaining_claude06_core_hour_caps=projection['remaining_caps'],reserve_after_current_plan_caps=3000-projection['actual_plus_remaining_caps'])
        save(path,b)
    s=json.loads((C/'REVIEW_STATUS.json').read_text());s.update(active_current_job_ids=[r['slurm_job_id'] for *_,results in sets for r in results if r['state'] not in TERMINAL],
         claude06_failures=failures,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         cumulative_actual_core_hours=json.loads((C/'BUDGET.json').read_text())['cumulative_actual_core_hours'],
         actual_gpu_card_hours=json.loads((G/'GPU_BUDGET.json').read_text())['actual_card_hours'])
    if failures:s.update(further_submissions_allowed=False,gpu_further_submissions_allowed=False,p0_state='CLAUDE06_HALTED_NEW_FAILURE')
    save(C/'REVIEW_STATUS.json',s)
    print(json.dumps(dict(failures=failures,pools=[dict(host=host,submitted=len(results),states={state:sum(r['state']==state for r in results) for state in sorted({r['state'] for r in results})},
             completed_usable=sum(r['usable'] for r in results)) for home,host,root,manifest,data,results in sets]),indent=2))

if __name__=='__main__':main()
