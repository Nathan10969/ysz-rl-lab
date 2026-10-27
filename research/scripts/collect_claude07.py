"""Recover Review07 evidence without submitting or retrying any calculation.

The default SSH operations are read-only. --guard is the sole remote mutation:
it cancels pending scientific descendants, or pending jobs in an unsafe pool.
Healthy running allocations are never cancelled by this collector.
"""
import argparse
import csv
import datetime
import io
import json
import math
from pathlib import Path
import re
import tarfile
import xml.etree.ElementTree as ET

import numpy as np
from ase.io import read

from claude07_safety import (TERMINAL, POOL_STOP, atomic_json, blocked_jobs,
                             budget_status, last_complete_calculation, validate_output)
from submit_claude05 import ssh

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = 'REVIEW_CLAUDE_07_gates.md'
POOLS = {
    'CPU': dict(home='calibration/claude07', host='hzw-cpu',
                remote='/online1/ycsc_hzw/hd100371/Jony/runs/02_calib_c07',
                historical=450.02666666666676, pool_limit=3000, round_limit=32, unit='core_hours'),
    'DFT_GPU': dict(home='calibration_gpu/claude07', host='your-cluster',
                    remote='/CLUSTER_HOME/runs/02_calib_c07',
                    historical=10.538055555555555, pool_limit=40, round_limit=27.5, unit='card_hours'),
}
FILES = ['OUTCAR', 'OSZICAR', 'CONTCAR', 'vasprun.xml', 'runtime_input_hashes.sha256',
         'RESULT_CHECK.json', 'TASK_STATUS.json', 'gpu_samples.csv', 'log.txt',
         'submitted_job_id.txt', 'INCAR', 'KPOINTS', 'POSCAR', 'allocation.json']


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load(path, default):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def read_marker(path):
    """Presence, including empty or malformed JSON, always means active."""
    path = Path(path)
    if not path.exists():
        return None
    raw = path.read_text(errors='replace')
    result = dict(present=True, raw_text=raw)
    try:
        result['data'] = json.loads(raw)
    except ValueError as error:
        result['parse_error'] = str(error)
    return result


def state_name(value):
    return (str(value or 'UNKNOWN').split()[0].split('+')[0]
            if str(value or 'UNKNOWN').split() else 'UNKNOWN')


def account_job(job, parent):
    """Bill only the allocation parent; steps are retained as diagnostics."""
    anomalies = []
    def number(key):
        try:
            value = float(parent.get(key) or 0)
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid number')
            return value
        except (ValueError, TypeError):
            anomalies.append('invalid_accounting:'+key)
            return 0.0
    elapsed = number('ElapsedRaw')
    cpu_time = number('CPUTimeRAW')
    allocated_cpus = number('AllocCPUS')
    match = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', parent.get('AllocTRES', ''))
    allocated_gpus = int(match.group(1)) if match else None
    if elapsed > 0:
        if allocated_cpus != job['cores']:
            anomalies.append('unexpected_AllocCPUS')
        if job['gpus'] and allocated_gpus != job['gpus']:
            anomalies.append('unexpected_or_unverified_AllocTRES_gpu')
        if not job['gpus'] and allocated_gpus not in (None, 0):
            anomalies.append('unexpected_GPU_on_CPU_pool')
        if not job['gpus'] and not parent.get('CPUTimeRAW'):
            anomalies.append('missing_CPUTimeRAW')
    actual = ((allocated_gpus if allocated_gpus is not None else job['gpus'])*elapsed/3600
              if job['gpus'] else cpu_time/3600)
    return dict(actual=actual, parent_elapsed_seconds=elapsed,
                allocated_gpus=allocated_gpus, allocation_anomalies=anomalies,
                billing_basis='parent_ElapsedRaw_x_AllocTRES_GPU' if job['gpus'] else 'parent_CPUTimeRAW')


def parse_account(data):
    lines = data.get('account', {}).get('stdout', '').splitlines()
    headers = lines[0].split('|') if lines else []
    rows = [dict(zip(headers, line.split('|'))) for line in lines[1:] if line.strip()]
    ids = {str(j['slurm_job_id']) for j in data['ledger']['jobs']}
    data['parents'] = {r['JobIDRaw']: r for r in rows if r.get('JobIDRaw') in ids}
    data['steps'] = [r for r in rows if '.' in r.get('JobIDRaw', '')]
    return data


def fetch_snapshot(home, settings, manifest):
    local = load(home/'SUBMISSIONS.json', {'jobs': []})
    request = dict(remote=settings['remote'], local=local, manifest=manifest)
    code = '''import datetime,json,os,subprocess
from pathlib import Path
request=REQUEST
root=Path(request['remote'])
def marker(path):
    if not path.exists():return None
    raw=path.read_text(errors='replace');r={'present':True,'raw_text':raw}
    try:r['data']=json.loads(raw)
    except ValueError as error:r['parse_error']=str(error)
    return r
def run(args):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=35)
    return {'code':p.returncode,'stdout':p.stdout.decode('utf8','replace'),'stderr':p.stderr.decode('utf8','replace')}
remote_ledger=root/'SUBMISSIONS.json'
ledger=json.loads(remote_ledger.read_text()) if remote_ledger.exists() else request['local']
ids=','.join(str(j['slurm_job_id']) for j in ledger['jobs'])
result={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'ledger':ledger,'ledger_source':'remote' if remote_ledger.exists() else 'local_query_fallback',
 'account':run(['sacct','-j',ids,'-P','--units=K','-o','JobIDRaw,JobName%80,State,ExitCode,ElapsedRaw,AllocCPUS,NTasks,CPUTimeRAW,AllocTRES%150,MaxRSS,NodeList,Start,End']) if ids else {'code':0,'stdout':'','stderr':''},
 'queue':run(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j|%T|%C|%b|%R']),
 'stop':marker(root/'CLAUDE07_POOL_STOP.json'),'files':{},'task_status':{}}
meta={j['name']:j for j in request['manifest']}
for job in ledger['jobs']:
    d=Path(meta.get(job['name'],job).get('remote_directory',str(root/job['name'])))
    result['files'][job['name']]={}
    result['task_status'][job['name']]=marker(d/'TASK_STATUS.json')
    for name in ['OSZICAR','OUTCAR','log.txt','slurm-'+str(job['slurm_job_id'])+'.out']:
        p=d/name
        if p.is_file():
            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-12000));tail=f.read().decode('utf8','replace')
            result['files'][job['name']][name]={'bytes':p.stat().st_size,'tail':tail}
print(json.dumps(result))
'''.replace('REQUEST', repr(request))
    result = ssh(settings['host'], 'python3 -', code.encode(), timeout=120)
    data = json.loads(result.stdout)
    (home/'evidence').mkdir(parents=True, exist_ok=True)
    if data['account']['code'] or data['queue']['code']:
        atomic_json(home/'evidence/QUERY_ERROR.json', data)
        raise RuntimeError('Read-only Slurm query failed; evidence saved in '+str(home/'evidence/QUERY_ERROR.json'))
    remote_ids = {j['name']: str(j['slurm_job_id']) for j in data['ledger']['jobs']}
    if any(remote_ids.get(j['name']) != str(j['slurm_job_id']) for j in local['jobs']):
        atomic_json(home/'evidence/LEDGER_CONFLICT.json', data)
        raise ValueError('Local/remote submission ledger conflict; no control action taken')
    unknown = set(remote_ids)-{j['name'] for j in manifest}
    if unknown:
        atomic_json(home/'evidence/LEDGER_CONFLICT.json', data)
        raise ValueError('Unrecognized remote jobs: '+str(sorted(unknown)))
    # Remote-only IDs are recovered, never resubmitted.
    if data['ledger_source'] == 'remote':
        atomic_json(home/'SUBMISSIONS.json', data['ledger'])
    atomic_json(home/'LATEST.json', data)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    atomic_json(home/'evidence'/('snapshot_'+stamp+'.json'), data)
    return parse_account(data)


def retrieve(home, settings, manifest, data):
    output = home/'outputs'
    output.mkdir(parents=True, exist_ok=True)
    byname = {j['name']: j for j in manifest}
    wanted = []
    for attempt in data['ledger']['jobs']:
        sid = str(attempt['slurm_job_id'])
        state = state_name(data['parents'].get(sid, {}).get('State'))
        if state not in TERMINAL:
            continue
        marker = output/attempt['name']/'RETRIEVED_TERMINAL.json'
        if load(marker, {}).get('state') == state:
            continue
        wanted.append(dict(name=attempt['name'], state=state, slurm_job_id=sid,
                           remote_directory=byname[attempt['name']]['remote_directory'],
                           files=FILES+['slurm-'+sid+'.out']))
    if not wanted:
        return
    code = '''import sys,tarfile
from pathlib import Path
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
    for job in JOBS:
        for name in job['files']:
            p=Path(job['remote_directory'])/name
            if p.is_file():tar.add(str(p),arcname=job['name']+'/'+name,recursive=False)
'''.replace('JOBS', repr(wanted))
    result = ssh(settings['host'], 'python3 -', code.encode(), timeout=180)
    names = {j['name'] for j in wanted}
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:') as tar:
        for item in tar:
            parts = Path(item.name).parts
            if len(parts) != 2 or parts[0] not in names or parts[1] not in FILES+['slurm-'+j['slurm_job_id']+'.out' for j in wanted] or not item.isfile():
                raise ValueError('Unsafe recovered archive member: '+item.name)
            dest = output/item.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(item).read())
    for job in wanted:
        directory = output/job['name']
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory/'RETRIEVED_TERMINAL.json', dict(state=job['state'],
                    at_utc=data['checked_at_utc'], files_present=sorted(p.name for p in directory.iterdir() if p.is_file())))


def geometry_matches(frame, xyz):
    atoms = read(xyz)
    if frame['symbols'] != atoms.get_chemical_symbols() or not np.allclose(frame['basis'], atoms.cell.array, atol=1e-6):
        return False
    positions = np.asarray(frame['positions'])
    if positions.shape != (len(atoms), 3):
        return False
    delta = positions-atoms.get_scaled_positions(wrap=False)
    delta -= np.rint(delta)
    return bool(np.max(np.linalg.norm(researcher@example.org, axis=1)) < 1e-6)


def analyze_one(root, home, job, attempt, data):
    sid = str(attempt['slurm_job_id'])
    parent = data['parents'].get(sid, {})
    state = state_name(parent.get('State'))
    result = dict(job, slurm_job_id=sid, state=state, accounting=parent,
                  usable=False, task_status=data.get('task_status', {}).get(job['name']),
                  step_accounting=[r for r in data['steps'] if r['JobIDRaw'].startswith(sid+'.')])
    result.update(account_job(job, parent))
    tails = data.get('files', {}).get(job['name'], {})
    all_tails = '\n'.join(v['tail'] for v in tails.values())
    result['tail_BRMIX_serious_count'] = all_tails.count('BRMIX: very serious problems')
    scf = re.findall(r'^\s*(?:DAV|RMM|CG):\s*(\d+)\s+', tails.get('OSZICAR', {}).get('tail', ''), re.M)
    result['last_printed_SCF_step'] = int(scf[-1]) if scf else None
    directory = home/'outputs'/job['name']
    result['terminal_outputs_retrieved'] = (directory/'RETRIEVED_TERMINAL.json').exists()
    if state not in TERMINAL or not (directory/'vasprun.xml').exists():
        if state == 'COMPLETED':
            result['validation_pending_retrieval'] = True
        return result
    try:
        xml = (directory/'vasprun.xml').read_text(errors='replace')
        out = (directory/'OUTCAR').read_text(errors='replace') if (directory/'OUTCAR').exists() else ''
        result['convergence'] = validate_output(xml, out, job['kind'] == 'relax')
        frame = last_complete_calculation(xml)
        if len(frame['positions']) != job['n_atoms'] or len(frame['symbols']) != job['n_atoms']:
            raise ValueError('Final complete frame atom count mismatch')
        forces = np.asarray(frame['forces'])
        result['final_geometry'] = dict(symbols=frame['symbols'], cell_A=frame['basis'],
                                        scaled_positions=frame['positions'], pbc=[True]*3)
        result['last_complete_frame'] = {k: v for k, v in frame.items() if k not in ['calculation_xml', 'positions', 'basis', 'forces', 'symbols']}
        energy = ET.fromstring(frame['calculation_xml'])
        result['energy_eV'] = {i.attrib['name']: float(i.text) for i in energy.findall('energy/i')}
        if not result['energy_eV'] or not all(math.isfinite(v) for v in result['energy_eV'].values()):
            raise ValueError('Missing or invalid energy')
        result['forces_eV_A'] = forces.tolist()
        result['max_force_eV_A'] = float(np.linalg.norm(forces, axis=1).max())
        result['SCF_loop_real_seconds'] = [float(x) for x in re.findall(r'LOOP:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)', out)]
        result['usable'] = bool(state == 'COMPLETED' and parent.get('ExitCode') == '0:0'
                                and result['convergence']['valid'] and not result['allocation_anomalies'])
        if job.get('mace_geometry'):
            # For relaxations check the submitted geometry, not the changed final frame.
            if job['kind'] == 'sp':
                result['input_geometry_matches_mace'] = geometry_matches(frame, root/job['mace_geometry'])
            else:
                poscar = directory/'POSCAR'
                if not poscar.exists():
                    poscar = home/'execution'/job['name']/'POSCAR'
                atoms = read(poscar, format='vasp')
                initial = dict(symbols=atoms.get_chemical_symbols(), basis=atoms.cell.array,
                               positions=atoms.get_scaled_positions(wrap=False))
                result['input_geometry_matches_mace'] = geometry_matches(initial, root/job['mace_geometry'])
    except Exception as error:
        result.update(usable=False, parse_error=repr(error))
    return result


def guard_plan(jobs, results, budget, stop):
    failures = set()
    status_errors = []
    for row in results:
        marker = row.get('task_status') or {}
        status = marker.get('data')
        if marker.get('present') and not isinstance(status, dict):
            status_errors.append(dict(name=row['name'], marker=marker))
            # Unknown task evidence prevents dependent science from starting;
            # it does not justify stopping an independent resource pool.
            failures.add(row['name'])
        if ((row['state'] in TERMINAL and row['state'] != 'COMPLETED') or
                (row['state'] == 'COMPLETED' and not row['usable'] and not row.get('validation_pending_retrieval')) or
                (isinstance(status, dict) and status.get('state') in ['FAILED', 'BLOCKED'])):
            failures.add(row['name'])
    descendants = blocked_jobs(jobs, failures)
    reasons = list(budget['reasons'])
    for row in results:
        reasons.extend(row['name']+':'+reason for reason in row.get('allocation_anomalies', []))
    pool_stop = bool(reasons) or bool(stop and stop.get('present'))
    pending = {r['name'] for r in results if r['state'] == 'PENDING'}
    cancel = [j['name'] for j in jobs if j['name'] in pending and (pool_stop or j['name'] in descendants)]
    return dict(failed_tasks=sorted(failures), blocked_descendants=descendants,
                pool_stop_required=pool_stop, pool_stop_reasons=reasons,
                existing_stop=stop, cancel_names=cancel, never_cancel_running=True,
                task_status_errors=status_errors)


def apply_guard(home, settings, manifest, data, plan):
    """Recheck live queue immediately before cancelling only pending own IDs."""
    byname = {j['name']: j for j in manifest}
    byid = {j['name']: str(j['slurm_job_id']) for j in data['ledger']['jobs']}
    request = dict(remote=settings['remote'], plan=plan,
                   targets={name: byid[name] for name in plan['cancel_names']},
                   task_directories={name: byname[name]['remote_directory'] for name in plan['blocked_descendants'] if name in byid})
    code = '''import datetime,json,os,subprocess,tempfile
from pathlib import Path
request=REQUEST
root=Path(request['remote']);plan=request['plan']
def atomic(path,value):
    with tempfile.NamedTemporaryFile(mode='w',dir=str(path.parent),prefix='.'+path.name+'.',delete=False) as f:
        json.dump(value,f,indent=2);f.write('\\n');f.flush();os.fsync(f.fileno());temporary=f.name
    os.replace(temporary,str(path))
def atomic_if_absent(path,value):
    # Publish a fully flushed file without ever replacing an existing marker.
    with tempfile.NamedTemporaryFile(mode='w',dir=str(path.parent),prefix='.'+path.name+'.',delete=False) as f:
        json.dump(value,f,indent=2);f.write('\\n');f.flush();os.fsync(f.fileno());temporary=f.name
    try:
        try:os.link(temporary,str(path))
        except FileExistsError:pass
    finally:os.unlink(temporary)
stop=root/'CLAUDE07_POOL_STOP.json'
if plan['pool_stop_required'] and not stop.exists():
    atomic_if_absent(stop,{'scope':'this_resource_pool_only','reasons':plan['pool_stop_reasons'],
                          'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
q=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i|%T']).decode()
pending={line.split('|')[0] for line in q.splitlines() if line.split('|')[1]=='PENDING'}
cancelled=[];started=[];unconfirmed=[]
for name,sid in request['targets'].items():
    if sid not in pending:continue
    # --state=PENDING closes the snapshot/cancel race: an allocation which
    # started after squeue must remain running under its original cap.
    subprocess.check_call(['scancel','--state=PENDING',sid])
    after=subprocess.run(['squeue','-h','-j',sid,'-o','%T'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    states=[line.strip().split('|')[-1] for line in after.stdout.decode().splitlines() if line.strip()]
    if after.returncode or 'PENDING' in states:unconfirmed.append(sid);continue
    if states:started.append(sid);continue
    cancelled.append(sid)
    if name in request['task_directories']:
        p=Path(request['task_directories'][name])/'TASK_STATUS.json'
        if not p.exists():atomic_if_absent(p,{'state':'BLOCKED','reason':'failed_scientific_dependency',
                                             'failed_tasks':plan['failed_tasks'],'slurm_job_id':sid})
print(json.dumps({'cancelled_only_pending_own_ids':cancelled,'running_jobs_cancelled':False,
                  'started_before_cancellation':started,'cancellation_not_yet_confirmed':unconfirmed,
                  'pool_stop_present':stop.exists(),'existing_stop_preserved':True}))
'''.replace('REQUEST', repr(request))
    result = ssh(settings['host'], 'python3 -', code.encode(), timeout=90)
    applied = json.loads(result.stdout)
    atomic_json(home/'evidence/GUARD_LAST.json', dict(plan=plan, applied=applied, at_utc=utcnow()))
    if plan['pool_stop_required']:
        marker = home/'execution'/POOL_STOP
        if not marker.exists():
            atomic_json(marker, plan['existing_stop'] or dict(scope='this_resource_pool_only', reasons=plan['pool_stop_reasons']))
    return applied


def collect_pool(root, name, settings, args):
    home = root/settings['home']
    manifest = load(home/'JOB_LIST.json', [])
    if isinstance(manifest, dict):
        manifest = manifest['jobs']
    if not manifest:
        raise ValueError('Missing prepared manifest: '+str(home/'JOB_LIST.json'))
    if args.from_snapshots:
        data = parse_account(load(home/'LATEST.json', {}))
    else:
        data = fetch_snapshot(home, settings, manifest)
    if args.retrieve:
        retrieve(home, settings, manifest, data)
    byname = {j['name']: j for j in manifest}
    rows = [analyze_one(root, home, byname[a['name']], a, data) for a in data['ledger']['jobs']]
    observed = {r['name']: dict(state=r['state'], actual=r['actual']) for r in rows}
    budget = budget_status(manifest, observed, settings['historical'], settings['pool_limit'], settings['round_limit'])
    budget.update(pool=name, unit=settings['unit'], historical_frozen_actual=settings['historical'],
                  review07_actual=sum(r['actual'] for r in rows),
                  authorized_remaining=settings['pool_limit']-budget['actual'],
                  approval_reference=REFERENCE, accounting_updated_at_utc=data['checked_at_utc'],
                  billing='Parent allocation only. Step rows are diagnostics; no duplicate charge.')
    plan = guard_plan(manifest, rows, budget, data.get('stop') or read_marker(home/'execution'/POOL_STOP))
    if plan['pool_stop_reasons']:
        budget.update(safe=False, reasons=plan['pool_stop_reasons'])
    atomic_json(home/'BUDGET.json', budget)
    atomic_json(home/'ANALYSIS.json', dict(checked_at_utc=data['checked_at_utc'], pool=name, jobs=rows, guard_plan=plan))
    with (home/'SCF_TIMINGS.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['name', 'slurm_job_id', 'SCF_step_global', 'wall_seconds_rank0'])
        for row in rows:
            for i, seconds in enumerate(row.get('SCF_loop_real_seconds', []), 1):
                writer.writerow([row['name'], row['slurm_job_id'], i, seconds])
    if args.guard and (plan['cancel_names'] or plan['pool_stop_required']):
        apply_guard(home, settings, manifest, data, plan)
    return dict(pool=name, submitted=len(rows), guard_requested=args.guard, guard_plan=plan, budget=budget)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--pools', nargs='+', choices=sorted(POOLS), default=list(POOLS))
    parser.add_argument('--retrieve', action='store_true', help='Recover outputs of all terminal jobs, including failures')
    parser.add_argument('--guard', action='store_true', help='Apply the recorded pending-only cancellation plan')
    parser.add_argument('--from-snapshots', action='store_true', help='Use existing local LATEST.json without querying SSH')
    args = parser.parse_args(argv)
    if args.from_snapshots and (args.retrieve or args.guard):
        parser.error('--from-snapshots is offline and cannot be combined with --retrieve or --guard')
    summaries = [collect_pool(args.root, name, POOLS[name], args) for name in args.pools]
    print(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
