"""Read-only SSH recovery for the independently budgeted Review07 smoke pool.

No model imports, Slurm submissions, preparation, or retries. --guard only
creates STOP_SMOKE for demonstrated budget/allocation/consumption risk and
cancels these two submitted jobs while they are still PENDING.
"""
import argparse
import copy
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

from claude07_safety import TERMINAL, atomic_json
from smoke_collect import ACTIVE, gpu_count
from submit_claude05 import ssh

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {'c07_smoke_sft': 'sft', 'c07_smoke_mace': 'mace'}
STOP = 'STOP_SMOKE.json'
OUTPUT_FILES = {'result.json', 'steps.jsonl', 'generations.json', 'ef_calls.jsonl', 'ef_attempts.jsonl',
                'relaxations.json', 'gpu_samples.csv', 'start.txt', 'end.txt',
                'slurm_allocation.txt', 'ADAPTER_METADATA.json'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def state(value):
    return str(value or 'ACCOUNTING_UNKNOWN').split()[0].rstrip('+')


def read_marker(path):
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


def validate_manifest(manifest, budget):
    if not isinstance(manifest, dict) or manifest.get('schema') != 'ysz.training_smoke.v1':
        raise ValueError('Unknown/missing smoke manifest schema')
    jobs = manifest.get('jobs', [])
    if len(jobs) != 2 or {j.get('id') for j in jobs} != set(EXPECTED):
        raise ValueError('Exactly the two reviewed smoke jobs are required')
    if not Path(manifest.get('remote_smoke_root', '')).is_absolute():
        raise ValueError('An absolute remote_smoke_root is required')
    caps = {'c07_smoke_sft': .5, 'c07_smoke_mace': .75}
    for job in jobs:
        if job.get('gpus') != 1 or job.get('cpu_cores') != 6 or job.get('hard_gpu_hours') != caps[job['id']]:
            raise ValueError('Smoke allocation or reviewed cap changed: '+job['id'])
    if not isinstance(budget, dict) or budget.get('authorized_card_hours') != 10:
        raise ValueError('Independent 10 card-hour smoke budget is required')
    budget_jobs = budget.get('jobs', [])
    if len(budget_jobs) != 2 or {j.get('id') for j in budget_jobs} != set(EXPECTED):
        raise ValueError('Smoke budget job schema is incomplete or duplicated')
    for job in budget_jobs:
        if job.get('gpus') != 1 or job.get('hard_gpu_hours') != caps[job['id']]:
            raise ValueError('Manifest/budget allocation or cap mismatch')


def reconcile_ledgers(manifest, local, remote):
    allowed = {j['id'] for j in manifest['jobs']}
    def indexed(ledger):
        if not isinstance(ledger, dict) or not isinstance(ledger.get('jobs'), list):
            raise ValueError('Unknown/missing submission ledger schema')
        rows = {}
        ids = set()
        for row in ledger['jobs']:
            if (not isinstance(row, dict) or row.get('id') not in allowed
                    or not re.fullmatch(r'[1-9][0-9]*', str(row.get('slurm_job_id', '')))
                    or row['id'] in rows or str(row['slurm_job_id']) in ids):
                raise ValueError('Unrecognized, duplicate, or invalid smoke submission')
            rows[row['id']] = row
            ids.add(str(row['slurm_job_id']))
        return rows
    local_rows = indexed(local)
    if remote is None:
        # Query known local IDs, but never describe a missing remote ledger as reconciled.
        return dict(local, ledger_status='REMOTE_LEDGER_UNKNOWN; local IDs queried only')
    remote_rows = indexed(remote)
    if any(name not in remote_rows or str(row['slurm_job_id']) != str(remote_rows[name]['slurm_job_id'])
           for name, row in local_rows.items()):
        raise ValueError('Local/remote smoke submission ID conflict')
    return dict(remote, ledger_status='RECONCILED', jobs=[dict(row, slurm_job_id=str(row['slurm_job_id']))
                for row in remote['jobs']])


def query(home, manifest, host):
    local = load(home/'SUBMISSIONS.json', {'jobs': []})
    previous_snapshot = load(home/'LATEST.json', {})
    reconcile_ledgers(manifest, local, None)
    request = dict(remote=manifest['remote_smoke_root'], local=local)
    code = '''import datetime,json,os,subprocess
from pathlib import Path
request=REQUEST;root=Path(request['remote'])
def marker(path):
    if not path.exists():return None
    raw=path.read_text(errors='replace');r={'present':True,'raw_text':raw}
    try:r['data']=json.loads(raw)
    except ValueError as error:r['parse_error']=str(error)
    return r
def run(args):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=35)
    return {'code':p.returncode,'stdout':p.stdout.decode('utf8','replace'),'stderr':p.stderr.decode('utf8','replace')}
ledger_marker=marker(root/'SUBMISSIONS.json')
remote_ledger=ledger_marker.get('data') if ledger_marker else None
ledger=remote_ledger if isinstance(remote_ledger,dict) and isinstance(remote_ledger.get('jobs'),list) else request['local']
ids=[str(j.get('slurm_job_id','')) for j in ledger['jobs'] if isinstance(j,dict)]
if any(not i.isdigit() for i in ids):raise ValueError('Invalid remote Slurm ID; query rejected')
result={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'remote_ledger':remote_ledger,'ledger_marker':ledger_marker,
 'account':run(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobIDRaw,State,ElapsedRaw,AllocTRES%160,ExitCode,AllocCPUS,NTasks']) if ids else {'code':0,'stdout':'','stderr':''},
 'queue':run(['squeue','-u',os.environ['USER'],'-h','-o','%i|%T']),
 'stop':marker(root/'STOP_SMOKE.json'),'allocation_proofs':{}}
if result['account']['code']==0:
    for line in result['account']['stdout'].splitlines():
        fields=line.strip().split('|')
        if (len(fields)>=7 and fields[0] in ids and fields[2].isdigit() and int(fields[2])>0
                and not fields[6].strip() and fields[0] not in result['allocation_proofs']):
            # This fills one missing field during the original observation;
            # it is not a retry of sacct, a job, or a model invocation.
            sid=fields[0];proof=run(['scontrol','show','job',sid,'-o'])
            proof['command']=['scontrol','show','job',sid,'-o']
            result['allocation_proofs'][sid]=proof
print(json.dumps(result))
'''.replace('REQUEST', repr(request))
    result = ssh(host, 'python3 -', code.encode(), timeout=120)
    snapshot = json.loads(result.stdout)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    snapshot_path = home/'evidence'/('snapshot_'+stamp+'.json')
    for sid, proof in snapshot.get('allocation_proofs', {}).items():
        proof['source'] = str(snapshot_path)+'#allocation_proofs.'+sid
        proof['captured_at_utc'] = snapshot['checked_at_utc']
        if isinstance(proof.get('stdout'), str):
            proof['stdout_sha256'] = hashlib.sha256(proof['stdout'].encode('utf8')).hexdigest()
    snapshot = preserve_terminal_allocation_proofs(snapshot, previous_snapshot, manifest)
    atomic_json(home/'LATEST.json', snapshot)
    atomic_json(snapshot_path, snapshot)
    remote_marker = snapshot.get('ledger_marker')
    if remote_marker and not isinstance(remote_marker.get('data'), dict):
        raise ValueError('Remote SUBMISSIONS.json is malformed; raw evidence retained in LATEST.json')
    ledger = reconcile_ledgers(manifest, local, snapshot.get('remote_ledger'))
    snapshot['ledger'] = ledger
    return snapshot


def ntasks_allocation_proof(proof, sid, job, sacct_tres):
    """Verify raw same-parent evidence; never synthesize sacct columns."""
    def rejected(reason, resource_mismatch=False):
        return dict(valid=False, reason=reason, resource_mismatch=resource_mismatch)
    if not isinstance(proof, dict):
        return rejected('missing_NTasks_allocation_proof')
    raw = proof.get('stdout')
    source = proof.get('source')
    if proof.get('code') != 0 or not isinstance(raw, str) or not raw.strip() or not isinstance(source, str) or not source:
        return rejected('invalid_NTasks_allocation_proof_source_or_query')
    digest = hashlib.sha256(raw.encode('utf8')).hexdigest()
    if proof.get('stdout_sha256', digest) != digest:
        return rejected('NTasks_allocation_proof_hash_mismatch')
    required = ['JobId', 'NumCPUs', 'NumTasks', 'CPUs/Task', 'TRES']
    matches = re.findall(r'(?:^|\s)(JobId|NumCPUs|NumTasks|CPUs/Task|TRES)=([^\s]+)', raw)
    if any(sum(key == name for key, value in matches) != 1 for name in required):
        return rejected('missing_or_duplicate_NTasks_allocation_proof_fields')
    parsed = dict(matches)
    if parsed['JobId'] != sid:
        return rejected('NTasks_allocation_proof_job_id_mismatch')
    if not all(parsed[key].isdigit() for key in ['NumCPUs', 'NumTasks', 'CPUs/Task']):
        return rejected('invalid_NTasks_allocation_proof_resources')
    if (parsed['NumCPUs'] != str(job['cpu_cores']) or parsed['NumTasks'] != '1'
            or parsed['CPUs/Task'] != str(job['cpu_cores'])):
        return rejected('NTasks_allocation_proof_resource_mismatch', True)
    entries = parsed['TRES'].split(',')
    if any('=' not in entry for entry in entries):
        return rejected('invalid_NTasks_allocation_proof_TRES')
    pairs = [entry.split('=', 1) for entry in entries]
    if len({key for key, value in pairs}) != len(pairs):
        return rejected('duplicate_NTasks_allocation_proof_TRES')
    tres = dict(pairs)
    gpu_values = [value for key, value in pairs if key == 'gres/gpu' or key.startswith('gres/gpu:')]
    if not tres.get('cpu', '').isdigit() or not gpu_values or not all(value.isdigit() for value in gpu_values):
        return rejected('missing_or_invalid_NTasks_allocation_proof_CPU_GPU_TRES')
    if tres['cpu'] != str(job['cpu_cores']) or any(value != str(job['gpus']) for value in gpu_values):
        return rejected('NTasks_allocation_proof_TRES_resource_mismatch', True)
    # A conservative comparison also retains memory/node/billing and typed GPU
    # fields: two representations which disagree do not repair the blank field.
    if tres != sacct_tres:
        conflict = any(key in sacct_tres and sacct_tres[key] != value for key, value in tres.items())
        return rejected('NTasks_allocation_proof_TRES_differs_from_sacct', conflict)
    evidence = dict(source=source, stdout_sha256=digest, raw_stdout=raw, parsed_allocation=parsed,
                    scope='Only blank sacct NTasks for this allocation parent; no billing fields replaced')
    for key in ['captured_at_utc', 'cache_reuse']:
        if key in proof:
            evidence[key] = copy.deepcopy(proof[key])
    return dict(valid=True, evidence=evidence)


def allocation_parent_binding(snapshot, sid):
    """A cache is tied to one unchanged parent, never to an MPI step or ID alone."""
    account = snapshot.get('account', {})
    if account.get('code') != 0:
        return None
    rows = [line.strip().split('|') for line in account.get('stdout', '').splitlines()
            if line.strip().split('|')[0] == sid]
    if len(rows) != 1:
        return None
    fields = rows[0]
    if (len(fields) < 7 or not fields[2].isdigit() or int(fields[2]) <= 0
            or not fields[5].isdigit() or fields[6].strip()
            or not re.fullmatch(r'\d+:\d+', fields[4])):
        return None
    entries = fields[3].split(',')
    if any('=' not in entry for entry in entries):
        return None
    pairs = [entry.split('=', 1) for entry in entries]
    if len({key for key, value in pairs}) != len(pairs):
        return None
    return dict(JobIDRaw=sid, State=state(fields[1]), ElapsedRaw=int(fields[2]),
                AllocTRES=dict(pairs), ExitCode=fields[4], AllocCPUS=fields[5], NTasks='')


def preserve_terminal_allocation_proofs(snapshot, previous, manifest):
    """Reuse verified terminal evidence only when a fresh query is unavailable.

    The original proof and its bound parent survive a transient sacct failure.
    Current accounting is still required; cached evidence never supplies time,
    state, CPU/GPU allocation, or a replacement for a conflicting fresh proof.
    """
    snapshot = copy.deepcopy(snapshot)
    previous = previous if isinstance(previous, dict) else {}
    meta = {job['id']: job for job in manifest['jobs']}
    def submitted(observation):
        ledger = observation.get('remote_ledger') or observation.get('ledger')
        try:
            checked = reconcile_ledgers(manifest, {'jobs': []}, ledger)
            return {str(row['slurm_job_id']): row['id'] for row in checked['jobs']}
        except (ValueError, TypeError):
            return {}
    current_ids = submitted(snapshot)
    prior_ids = submitted(previous)
    cache = {}
    # Revalidate persisted cache entries rather than treating stored flags as proof.
    for sid, entry in previous.get('allocation_proof_cache', {}).items():
        if not isinstance(entry, dict) or entry.get('job_id') not in meta:
            continue
        binding = entry.get('parent_binding', {})
        if binding.get('JobIDRaw') != sid or binding.get('State') not in TERMINAL:
            continue
        checked = ntasks_allocation_proof(entry.get('proof'), sid, meta[entry['job_id']], binding.get('AllocTRES', {}))
        if checked['valid']:
            cache[sid] = copy.deepcopy(entry)
    # Support the first reuse of an existing snapshot captured before this cache
    # field existed, including the parent's saved 635 proof injection.
    for sid, proof in previous.get('allocation_proofs', {}).items():
        binding = allocation_parent_binding(previous, sid)
        job_id = prior_ids.get(sid)
        if not binding or binding['State'] not in TERMINAL or job_id not in meta:
            continue
        checked = ntasks_allocation_proof(proof, sid, meta[job_id], binding['AllocTRES'])
        if checked['valid']:
            retained = copy.deepcopy(proof)
            retained.setdefault('captured_at_utc', previous.get('checked_at_utc'))
            retained.setdefault('stdout_sha256', checked['evidence']['stdout_sha256'])
            cache[sid] = dict(job_id=job_id, parent_binding=binding, proof=retained,
                              bound_at_utc=previous.get('checked_at_utc'))
        elif isinstance(proof, dict) and proof.get('code') == 0:
            cache.pop(sid, None)
    incoming = copy.deepcopy(snapshot.get('allocation_proofs', {}))
    snapshot['fresh_allocation_proofs'] = copy.deepcopy(incoming)
    snapshot['allocation_proofs'] = copy.deepcopy(incoming)
    reused = []
    for sid, fresh in incoming.items():
        binding = allocation_parent_binding(snapshot, sid)
        job_id = current_ids.get(sid)
        if not isinstance(fresh, dict) or job_id not in meta:
            continue
        if fresh.get('code') == 0:
            # Fresh successful evidence always wins, including contradictions.
            checked = ntasks_allocation_proof(fresh, sid, meta[job_id], binding['AllocTRES']) if binding else {'valid': False}
            if binding and binding['State'] in TERMINAL and checked['valid']:
                retained = copy.deepcopy(fresh)
                retained.setdefault('captured_at_utc', snapshot.get('checked_at_utc'))
                retained.setdefault('stdout_sha256', checked['evidence']['stdout_sha256'])
                cache[sid] = dict(job_id=job_id, parent_binding=binding, proof=retained,
                                  bound_at_utc=snapshot.get('checked_at_utc'))
            else:
                cache.pop(sid, None)
            continue
        diagnostic = str(fresh.get('stdout', ''))+'\n'+str(fresh.get('stderr', ''))
        if re.search(r'(?:^|\s)(?:JobId|JobIDRaw|NumCPUs|NumTasks|CPUs/Task|TRES|AllocTRES|AllocCPUS|NTasks)=', diagnostic, re.I):
            # Resource fields in either stream are ambiguous fresh evidence;
            # a purge phrase elsewhere must never hide that contradiction.
            cache.pop(sid, None)
            continue
        purged = re.search(r'\b(?:invalid job id specified|unknown job(?: id)?(?:[: ]+\d+)?)\s*[.!]?\s*$',
                           diagnostic, re.I | re.M)
        infrastructure_error = re.search(r'\b(?:permission denied|authentication failed|command not found|'
            r'connection (?:timed out|refused)|network is unreachable|could not resolve|not authorized|access denied)\b', diagnostic, re.I)
        if (type(fresh.get('code')) is not int or fresh['code'] != 1
                or not isinstance(fresh.get('stdout'), str) or not isinstance(fresh.get('stderr'), str)
                or not purged or infrastructure_error):
            # Missing commands, authentication and transport errors are not
            # evidence of a purged job. Retain the bound cache, but do not use
            # it for this observation or release its reservation.
            continue
        retained = cache.get(sid)
        if not retained or not binding:
            continue
        if (binding['State'] not in TERMINAL or retained['job_id'] != job_id
                or retained['parent_binding'] != binding):
            cache.pop(sid, None)
            continue
        checked = ntasks_allocation_proof(retained['proof'], sid, meta[job_id], binding['AllocTRES'])
        if not checked['valid'] or binding['AllocCPUS'] != str(meta[job_id]['cpu_cores']):
            cache.pop(sid, None)
            continue
        selected = copy.deepcopy(retained['proof'])
        selected['cache_reuse'] = dict(same_terminal_parent_binding=True,
            reused_at_utc=snapshot.get('checked_at_utc'), original_captured_at_utc=selected.get('captured_at_utc'),
            parent_binding_sha256=hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest(),
            fresh_scontrol_failure=copy.deepcopy(fresh))
        snapshot['allocation_proofs'][sid] = selected
        reused.append(sid)
    snapshot['allocation_proof_cache'] = cache
    snapshot['allocation_proof_cache_reused_job_ids'] = reused
    return snapshot


def parse_accounting(text, manifest, ledger, previous, allocation_proofs=None):
    """Unknown allocation remains unknown, with its full reservation retained."""
    submitted = {str(j['slurm_job_id']): j for j in ledger['jobs']}
    meta = {j['id']: j for j in manifest['jobs']}
    known = {j['id']: j for j in previous['jobs']}
    records = {}
    risks = []
    unknown = set(submitted)
    allocation_proofs = allocation_proofs if isinstance(allocation_proofs, dict) else {}
    for line in text.splitlines():
        fields = line.strip().split('|')
        if not line.strip():
            continue
        if len(fields) < 5 or fields[0] not in submitted:
            unknown.add('UNRECOGNIZED_PARENT:'+fields[0])
            continue
        sid, raw_state, elapsed, tres, exit_code = fields[:5]
        if sid in records:
            records[sid]['verified'] = False
            records[sid]['actual_card_hours'] = None
            records[sid]['accounting_issues'].append('duplicate_parent_accounting')
            unknown.add(sid)
            continue
        job = meta[submitted[sid]['id']]
        row = dict(slurm_job_id=sid, state=state(raw_state), exit_code=exit_code,
                   alloc_tres=tres, verified=False, allocation_verified=False,
                   actual_card_hours=None, raw_fields=fields, accounting_issues=[])
        records[sid] = row
        try:
            seconds = int(elapsed)
            if seconds < 0:
                raise ValueError('negative elapsed')
            row['elapsed_seconds'] = seconds
            if row['state'] not in ACTIVE | TERMINAL:
                raise ValueError('unknown Slurm state')
            if not re.fullmatch(r'\d+:\d+', exit_code):
                raise ValueError('unknown exit code')
            allocation = dict(field.split('=', 1) for field in tres.split(',') if '=' in field)
            gpu_values = [int(v) for k, v in allocation.items() if k == 'gres/gpu' or k.startswith('gres/gpu:')]
            if seconds and not gpu_values:
                row['accounting_issues'].append('missing_GPU_allocation')
                continue
            if any(v < 0 for v in gpu_values):
                raise ValueError('negative GPU allocation')
            if len(set(gpu_values)) > 1:
                row['accounting_issues'].append('inconsistent_GPU_allocation')
                continue
            count = gpu_count(tres, 1) if gpu_values else 0
            if seconds and count != 1:
                risks.append('unexpected_GPU_allocation:'+sid)
            effective_tasks = fields[6] if len(fields) >= 7 else None
            if seconds:
                if len(fields) < 7 or not fields[5].isdigit():
                    row['accounting_issues'].append('missing_CPU_MPI_allocation')
                    continue
                if not effective_tasks.strip():
                    # Only NTasks may be absent. CPU/GPU allocation must
                    # already be complete, consistent, and reviewed in sacct.
                    if fields[5] != str(job['cpu_cores']) or count != job['gpus'] or allocation.get('cpu') != fields[5]:
                        row['accounting_issues'].append('NTasks_fallback_requires_complete_matching_sacct_CPU_GPU')
                        if fields[5] != str(job['cpu_cores']) or (allocation.get('cpu', '').isdigit() and allocation['cpu'] != fields[5]):
                            risks.append('unexpected_CPU_MPI_allocation:'+sid)
                        continue
                    checked = ntasks_allocation_proof(allocation_proofs.get(sid), sid, job, allocation)
                    if not checked['valid']:
                        row['accounting_issues'].append(checked['reason'])
                        if checked.get('resource_mismatch'):
                            risks.append('allocation_proof_resource_mismatch:'+sid)
                        continue
                    row['NTasks_fallback_evidence'] = checked['evidence']
                    effective_tasks = checked['evidence']['parsed_allocation']['NumTasks']
                elif not effective_tasks.isdigit():
                    row['accounting_issues'].append('missing_CPU_MPI_allocation')
                    continue
                if fields[5] != str(job['cpu_cores']) or effective_tasks != '1':
                    risks.append('unexpected_CPU_MPI_allocation:'+sid)
            cost = count*seconds/3600
            old = known[job['id']]
            if old.get('slurm_job_id') not in [None, sid, int(sid)]:
                raise ValueError('old billed ID changed')
            if cost+1e-9 < float(old.get('actual_card_hours', 0)):
                row['accounting_issues'].append('accounting_regression')
                continue
            row.update(actual_card_hours=cost, allocated_gpus=count, verified=True,
                       allocation_verified=bool(seconds > 0 and count == 1 and len(fields) >= 7
                                                and fields[5] == str(job['cpu_cores']) and effective_tasks == '1'))
            unknown.discard(sid)
            if cost > job['hard_gpu_hours']+1e-6 or cost > 2+1e-6:
                risks.append('per_job_card_hour_cap_exceeded:'+sid)
        except (ValueError, TypeError, OverflowError) as error:
            row['accounting_issues'].append('invalid_parent_accounting:'+str(error))
    return records, sorted(set(risks)), sorted(unknown)


def reconcile_budget(home, ledger, records, unknown, source):
    """Run the existing helper against an isolated copy; atomically publish once."""
    original = load(home/'BUDGET.json')
    submitted = {j['id']: str(j['slurm_job_id']) for j in ledger['jobs']}
    for job in original['jobs']:
        if job.get('slurm_job_id') and job['id'] not in submitted:
            raise ValueError('Prior billed run missing from submissions; budget left unchanged')
    verified_lines = []
    for sid, row in records.items():
        if row['verified'] and sid not in unknown:
            verified_lines.append('|'.join([sid, row['state'], str(row['elapsed_seconds']), row['alloc_tres'], row['exit_code']]))
    with tempfile.TemporaryDirectory(prefix='c07-smoke-account-') as tmp:
        scratch = Path(tmp)
        atomic_json(scratch/'BUDGET.json', original)
        atomic_json(scratch/'SUBMISSIONS.json', ledger)
        (scratch/'parents.tsv').write_text('\n'.join(verified_lines)+'\n')
        subprocess.run([sys.executable, str(ROOT/'scripts/smoke_collect.py'), '--smoke-home', str(scratch),
                        '--sacct-tsv', str(scratch/'parents.tsv')], check=True, capture_output=True, text=True)
        budget = load(scratch/'BUDGET.json')
    budget.update(last_sacct_source=source, accounting_updated_at_utc=now(),
                  formal_training_released=False, accounting_complete=not unknown,
                  billing='Only verified parent ElapsedRaw x allocated GPU count. Step rows never billed.',
                  independent_of_DFT_40_gpu_hour_pool=True, transfers_between_pools_permitted=False)
    if unknown:
        budget['actual_status'] = 'INCOMPLETE: actual is a known lower bound; missing records retain prior costs and full remaining caps'
    atomic_json(home/'BUDGET.json', budget)
    return budget


def finite_tree(value, path='result'):
    if isinstance(value, float) and not math.isfinite(value):
        return [path+' is nonfinite']
    if isinstance(value, dict):
        return [error for k, item in value.items() for error in finite_tree(item, path+'.'+str(k))]
    if isinstance(value, list):
        return [error for i, item in enumerate(value) for error in finite_tree(item, path+'['+str(i)+']')]
    return []


def json_evidence(value):
    """Keep invalid numeric evidence legible while emitting standards-valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return dict(invalid_nonfinite_numeric_value=repr(value), original_preserved_in='outputs/result.json')
    if isinstance(value, dict):
        return {key: json_evidence(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_evidence(item) for item in value]
    return value


def validate_result(job, result, sid):
    errors = []
    consumption = []
    if not isinstance(result, dict):
        return dict(valid=False, errors=['Unknown result schema'], consumption_risks=[])
    errors += finite_tree(result)
    if result.get('status') != 'completed':
        errors.append('result.status is not completed')
    runtime = result.get('runtime')
    if not isinstance(runtime, dict) or str(runtime.get('job_id')) != sid or runtime.get('visible_gpus') != 1:
        errors.append('Missing/mismatched runtime Slurm ID or visible GPU count')
    def metric(name, integer=False, positive=False):
        value = result.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or (integer and type(value) is not int) or value < 0 or (positive and value <= 0)):
            errors.append('Missing/invalid metric:'+name)
            return None
        return value
    metric('wall_seconds', positive=True)
    if job['id'] == 'c07_smoke_sft':
        seen = metric('seen_tokens', integer=True, positive=True)
        action = metric('action_tokens', integer=True, positive=True)
        steps = metric('optimizer_steps', integer=True, positive=True)
        if steps is not None and steps != job.get('optimization_steps', 32):
            errors.append('optimizer_steps differs from reviewed 32-step smoke')
        if steps is not None and steps > job.get('optimization_steps', 32):
            consumption.append('SFT_optimizer_step_cap_exceeded')
        if seen is not None and seen > job.get('seen_token_cap', 32768):
            consumption.append('SFT_seen_token_cap_exceeded')
        if seen is not None and action is not None and action > seen:
            errors.append('Action tokens exceed seen tokens')
        microbatches = metric('microbatches', integer=True, positive=True)
        validation_n = metric('validation_n', integer=True, positive=True)
        generation_tokens = metric('generation_tokens', integer=True)
        if microbatches != 64:
            errors.append('Expected 64 microbatches')
        if microbatches is not None and microbatches > 64:
            consumption.append('SFT_microbatch_cap_exceeded')
        if validation_n is not None and validation_n > 4:
            consumption.append('SFT_validation_prompt_cap_exceeded')
        if generation_tokens is not None and generation_tokens > 256:
            consumption.append('SFT_generation_token_cap_exceeded')
        reload = result.get('adapter_reload')
        if not isinstance(reload, dict) or any(reload.get(key) is not True for key in
                ['unloaded_without_merge', 'reloaded_from_disk', 'state_exact_match']):
            errors.append('Adapter unload/reload/exact-state checks missing or failed')
    elif job['id'] == 'c07_smoke_mace':
        values = {key: metric(key, integer=True, positive=True) for key in
                  ['raw_ef_calls', 'energy_calls', 'force_calls', 'model_forward_calls']}
        raw = values['raw_ef_calls']
        if raw is not None:
            if raw > job.get('raw_ef_call_cap', 1024):
                consumption.append('MACE_raw_EF_call_cap_exceeded')
            if values['energy_calls'] != raw or values['force_calls'] != raw:
                errors.append('Energy/force counters differ from raw E/F calls')
            if values['model_forward_calls'] is not None and values['model_forward_calls'] > raw:
                errors.append('Model forwards exceed raw cell E/F calls')
        throughput = result.get('single_batch_SP')
        sizes = [str(size) for size in job.get('batch_sizes', [1, 2, 4])]
        parity_sizes = [size for size in sizes if size != '1']
        if not isinstance(throughput, dict) or not set(sizes) <= set(throughput):
            errors.append('SP throughput missing required batch sizes '+ '/'.join(sizes))
        else:
            for size in sizes:
                value = throughput[size].get('cells_per_second') if isinstance(throughput[size], dict) else None
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    errors.append('Invalid cells_per_second for batch '+size)
        parity = result.get('batch_equivalence')
        pairs = parity.get('per_batch_size') if isinstance(parity, dict) else None
        if not isinstance(pairs, dict) or not set(parity_sizes) <= set(pairs):
            errors.append('Independent singleton/batch parity missing for '+ '/'.join(parity_sizes))
        else:
            for size in parity_sizes:
                pair = pairs[size]
                if not isinstance(pair, dict) or pair.get('passed') is not True:
                    errors.append('Parity check did not pass for batch '+size)
                    continue
                for key in ['max_energy_difference_eV', 'max_force_difference_eV_A']:
                    value = pair.get(key)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1e-6:
                        errors.append('Independent parity tolerance failed: batch '+size+' '+key)
        relaxations = result.get('relaxations')
        expected_relaxations = job.get('relaxation_cells', 6)
        if not isinstance(relaxations, list) or len(relaxations) != expected_relaxations:
            errors.append('Expected '+str(expected_relaxations)+' relaxation records')
            relaxations = relaxations if isinstance(relaxations, list) else []
        for row in relaxations:
            if not isinstance(row, dict) or type(row.get('steps')) is not int:
                errors.append('Invalid relaxation step schema')
            elif row['steps'] < 0:
                errors.append('Negative relaxation steps')
            elif row['steps'] > job.get('relax_steps_per_cell_cap', 40):
                consumption.append('MACE_relaxation_step_cap_exceeded')
    else:
        errors.append('Unknown smoke job ID')
    errors += consumption
    return dict(valid=not errors, errors=errors, consumption_risks=consumption)


def analyze_job(home, job, attempt, record):
    sid = str(attempt['slurm_job_id'])
    prefix = EXPECTED[job['id']]+'_'+sid
    output = home/'outputs'/prefix
    result = dict(id=job['id'], slurm_job_id=sid, state=record.get('state', 'ACCOUNTING_UNKNOWN'),
                  output_directory=str(output), terminal=record.get('state') in TERMINAL,
                  accounting=record,
                  accounting_verified=record.get('verified', False),
                  actual_card_hours=record.get('actual_card_hours'), usable=False,
                  terminal_outputs_retrieved=(output/'RETRIEVED_TERMINAL.json').exists(),
                  consumption_risks=[])
    path = output/'result.json'
    if path.exists():
        try:
            payload = load(path)
            check = validate_result(job, payload, sid)
            result.update(result=json_evidence(payload), result_check=check, consumption_risks=check['consumption_risks'])
            result['usable'] = bool(result['state'] == 'COMPLETED' and record.get('exit_code') == '0:0'
                                    and record.get('verified') and record.get('allocation_verified', record.get('verified'))
                                    and check['valid'])
        except (ValueError, TypeError) as error:
            result['result_error'] = repr(error)
    elif result['terminal']:
        result['result_error'] = 'Terminal job has no recovered result.json; cannot use throughput metrics'
    return result


def retrieve(home, manifest, host, ledger, records):
    wanted = []
    for job in ledger['jobs']:
        sid = str(job['slurm_job_id'])
        row = records.get(sid, {})
        if row.get('state') in TERMINAL:
            wanted.append(dict(id=job['id'], sid=sid, prefix=EXPECTED[job['id']]+'_'+sid,
                               kind=EXPECTED[job['id']], state=row['state']))
    if not wanted:
        return
    # The remote script only reads whitelisted outputs. Adapter weights stay
    # remote; their names, sizes, and SHA256 fingerprints are recovered instead.
    code = '''import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(REMOTE);wanted=JOBS;allowed=set(ALLOWED)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
    for job in wanted:
        output=root/'outputs'/job['prefix'];base='outputs/'+job['prefix']+'/'
        if output.is_dir():
            for p in output.iterdir():
                if p.is_file() and not p.is_symlink() and (p.name in allowed or (p.name.startswith('relax_') and p.suffix=='.extxyz')):
                    tar.add(str(p),arcname=base+p.name,recursive=False)
            adapter=output/'adapter';metadata=[]
            if adapter.is_dir():
                for p in sorted(adapter.rglob('*')):
                    if not p.is_file() or p.is_symlink():continue
                    h=hashlib.sha256()
                    with p.open('rb') as f:
                        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
                    metadata.append({'path':str(p.relative_to(adapter)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
                data=json.dumps({'adapter_files':metadata,'weights_transferred':False},indent=2).encode()
                info=tarfile.TarInfo(base+'ADAPTER_METADATA.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
        log=root/'jobs'/job['kind']/('slurm-'+job['sid']+'.out')
        if log.is_file() and not log.is_symlink():tar.add(str(log),arcname=base+log.name,recursive=False)
'''.replace('REMOTE', repr(manifest['remote_smoke_root'])).replace('JOBS', repr(wanted)).replace('ALLOWED', repr(sorted(OUTPUT_FILES)))
    response = ssh(host, 'python3 -', code.encode(), timeout=180)
    prefixes = {j['prefix'] for j in wanted}
    with tarfile.open(fileobj=io.BytesIO(response.stdout), mode='r:') as tar:
        for item in tar:
            parts = Path(item.name).parts
            if (len(parts) != 3 or parts[0] != 'outputs' or parts[1] not in prefixes or not item.isfile()
                    or not (parts[2] in OUTPUT_FILES or re.fullmatch(r'slurm-[0-9]+\.out', parts[2])
                            or (parts[2].startswith('relax_') and parts[2].endswith('.extxyz')))):
                raise ValueError('Unsafe smoke recovery archive member:'+item.name)
            destination = home/item.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(tar.extractfile(item).read())
    for job in wanted:
        directory = home/'outputs'/job['prefix']
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory/'RETRIEVED_TERMINAL.json', dict(at_utc=now(), state=job['state'],
                    files_present=sorted(p.name for p in directory.iterdir() if p.is_file()), weights_transferred=False))


def guard_plan(rows, budget, risks, stop):
    reasons = list(risks)
    if budget.get('actual_plus_remaining_caps', 0) > 10+1e-6:
        reasons.append('smoke_pool_card_hour_cap_exceeded')
    for row in budget.get('jobs', []):
        if row.get('actual_card_hours', 0) > min(row['hard_gpu_hours'], 2)+1e-6:
            reasons.append('per_job_card_hour_cap_exceeded:'+row['id'])
    for row in rows:
        reasons.extend(row['id']+':'+r for r in row.get('consumption_risks', []))
    required = bool(reasons) or bool(stop and stop.get('present'))
    return dict(pool='TRAINING_SMOKE', pool_stop_required=required,
                reasons=sorted(set(reasons)), existing_stop=stop,
                cancel_ids=[row['slurm_job_id'] for row in rows if row['state'] == 'PENDING'] if required else [],
                failed_tasks=[row['id'] for row in rows if row['state'] in TERMINAL and row['state'] != 'COMPLETED'],
                true_scientific_dependencies=[], never_cancel_running=True)


def apply_guard(home, manifest, host, ledger, plan):
    allowed = {str(j['slurm_job_id']) for j in ledger['jobs']}
    if set(plan['cancel_ids'])-allowed:
        raise ValueError('Guard refuses unrecognized Slurm IDs')
    request = dict(remote=manifest['remote_smoke_root'], ids=plan['cancel_ids'], plan=plan)
    code = '''import datetime,json,os,subprocess,tempfile
from pathlib import Path
request=REQUEST;root=Path(request['remote']);plan=request['plan'];stop=root/'STOP_SMOKE.json'
def absent_marker(path,data):
    with tempfile.NamedTemporaryFile(mode='w',dir=str(path.parent),prefix='.'+path.name+'.',delete=False) as f:
        json.dump(data,f,indent=2);f.write('\\n');f.flush();os.fsync(f.fileno());temporary=f.name
    try:
        try:os.link(temporary,str(path))
        except FileExistsError:pass
    finally:os.unlink(temporary)
if plan['pool_stop_required']:
    absent_marker(stop,{'scope':'TRAINING_SMOKE_ONLY','reasons':plan['reasons'],
                        'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
q=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i|%T']).decode()
pending={line.split('|')[0] for line in q.splitlines() if line.split('|')[-1]=='PENDING'}
requested=[]
for sid in request['ids']:
    if sid not in pending:continue
    subprocess.check_call(['scancel','--state=PENDING',sid]);requested.append(sid)
print(json.dumps({'pending_cancellation_requests':requested,'running_cancelled':False,
                  'stop_present':stop.exists(),'existing_stop_preserved':True}))
'''.replace('REQUEST', repr(request))
    response = ssh(host, 'python3 -', code.encode(), timeout=90)
    applied = json.loads(response.stdout)
    atomic_json(home/'evidence/GUARD_LAST.json', dict(at_utc=now(), plan=plan, applied=applied))
    # Mirror the stop evidence locally without replacing any existing raw marker.
    if plan['pool_stop_required'] and not (home/STOP).exists():
        atomic_json(home/STOP, plan['existing_stop'] or dict(scope='TRAINING_SMOKE_ONLY', reasons=plan['reasons']))
    return applied


def deferred_analysis(home, manifest, ledger, snapshot, records, unknown, errors, verified_risks):
    """A failed observation is not evidence of spending or a task failure."""
    meta = {j['id']: j for j in manifest['jobs']}
    rows = []
    for attempt in ledger.get('jobs', []):
        if attempt.get('id') not in meta:
            continue
        sid = str(attempt['slurm_job_id'])
        row = analyze_job(home, meta[attempt['id']], attempt, records.get(sid, {}))
        row.update(usable=False, validation_blocked_by='DEFERRED control/accounting observation')
        rows.append(row)
    analysis = dict(checked_at_utc=snapshot.get('checked_at_utc', now()),
                    collection_status='DEFERRED', pool='TRAINING_SMOKE', jobs=rows,
                    query_errors=errors, accounting_unknown_job_ids=unknown,
                    ledger_status=ledger.get('ledger_status', 'UNKNOWN'),
                    budget_untouched=True, submissions_untouched=True, guard_mutations_refused=True,
                    readonly_requery_required=True, readonly_requery_policy='Parent may perform one read-only retry; this script never retries jobs or observations.',
                    verified_risks_waiting_for_requery=verified_risks,
                    guard_plan=dict(pool='TRAINING_SMOKE', pool_stop_required=False, cancel_ids=[], reasons=[],
                                    deferred=True, existing_stop=snapshot.get('stop') or read_marker(home/STOP), never_cancel_running=True),
                    all_submitted_terminal=False, all_jobs_terminal=False,
                    all_terminal_outputs_retrieved=False, all_usable=False,
                    formal_training_released=False, no_retries=True)
    atomic_json(home/'evidence/QUERY_DEFERRED.json', dict(at_utc=now(), errors=errors, unknown=unknown,
                                                        verified_risks=verified_risks))
    atomic_json(home/'ANALYSIS.json', analysis)
    return analysis


def collect(home, host, do_retrieve=False, do_guard=False, offline=False):
    home = Path(home)
    manifest = load(home/'JOB_MANIFEST.json')
    previous = load(home/'BUDGET.json')
    validate_manifest(manifest, previous)
    if offline:
        snapshot = load(home/'LATEST.json')
        if not isinstance(snapshot, dict):
            raise ValueError('No saved smoke snapshot')
        snapshot['ledger'] = reconcile_ledgers(manifest, load(home/'SUBMISSIONS.json', {'jobs': []}), snapshot.get('remote_ledger'))
    else:
        try:
            snapshot = query(home, manifest, host)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
            # Keep every financial and execution control byte intact. The
            # parent can make its one authorized read-only observation retry.
            local = load(home/'SUBMISSIONS.json', {'jobs': []})
            return deferred_analysis(home, manifest, local, {}, {},
                [str(j['slurm_job_id']) for j in local.get('jobs', [])], [repr(error)], [])
    ledger = snapshot['ledger']
    query_errors = [key for key in ['account', 'queue'] if snapshot.get(key, {}).get('code') != 0]
    account_text = snapshot.get('account', {}).get('stdout', '') if 'account' not in query_errors else ''
    records, risks, unknown = parse_accounting(account_text, manifest, ledger, previous,
                                              allocation_proofs=snapshot.get('allocation_proofs'))
    queue = {}
    if 'queue' not in query_errors:
        for line in snapshot['queue']['stdout'].splitlines():
            fields = line.strip().split('|')
            if len(fields) == 2:
                queue[fields[0]] = state(fields[1])
    deferred_errors = ['query_failed:'+key for key in query_errors]
    if ledger.get('ledger_status') != 'RECONCILED' and ledger['jobs']:
        deferred_errors.append('remote_submission_ledger_unknown')
    if unknown or deferred_errors:
        return deferred_analysis(home, manifest, ledger, snapshot, records, unknown, deferred_errors, risks)
    if not offline and ledger.get('ledger_status') == 'RECONCILED':
        atomic_json(home/'SUBMISSIONS.json', ledger)
    source = 'training_smoke/LATEST.json at '+snapshot.get('checked_at_utc', 'UNKNOWN_TIME')
    budget = reconcile_budget(home, ledger, records, unknown, source)
    if do_retrieve:
        retrieve(home, manifest, host, ledger, records)
    meta = {j['id']: j for j in manifest['jobs']}
    rows = []
    for attempt in ledger['jobs']:
        sid = str(attempt['slurm_job_id'])
        record = records.get(sid, {})
        row = analyze_job(home, meta[attempt['id']], attempt, record)
        row['queue_state'] = queue.get(sid, 'NOT_SEEN')
        if row['state'] == 'ACCOUNTING_UNKNOWN' and queue.get(sid) == 'PENDING':
            row['state'] = 'PENDING'
            row['accounting_state'] = 'ACCOUNTING_UNKNOWN; pending reservation retained'
        rows.append(row)
    stop = snapshot.get('stop') or read_marker(home/STOP)
    plan = guard_plan(rows, budget, risks, stop)
    analysis = dict(checked_at_utc=snapshot.get('checked_at_utc'), collection_status='COLLECTED', pool='TRAINING_SMOKE', jobs=rows,
                    query_errors=query_errors, ledger_status=ledger.get('ledger_status'),
                    accounting_unknown_job_ids=unknown, guard_plan=plan,
                    all_submitted_terminal=len(rows) == 2 and all(row['terminal'] for row in rows),
                    all_jobs_terminal=len(rows) == 2 and all(row['terminal'] for row in rows),
                    all_terminal_outputs_retrieved=len(rows) == 2 and all(row['terminal'] and row['terminal_outputs_retrieved'] for row in rows),
                    all_usable=len(rows) == 2 and all(row['usable'] for row in rows),
                    formal_training_released=False, no_retries=True)
    atomic_json(home/'ANALYSIS.json', analysis)
    if do_guard and plan['pool_stop_required']:
        apply_guard(home, manifest, host, ledger, plan)
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-home', type=Path, default=ROOT/'training_smoke')
    parser.add_argument('--host', default='your-cluster')
    parser.add_argument('--retrieve', action='store_true')
    parser.add_argument('--guard', action='store_true')
    parser.add_argument('--from-snapshot', action='store_true', help='Offline accounting/analysis from local LATEST.json')
    args = parser.parse_args(argv)
    if args.from_snapshot and (args.retrieve or args.guard):
        parser.error('Offline snapshots cannot perform remote retrieval or guard actions')
    analysis = collect(args.smoke_home, args.host, args.retrieve, args.guard, args.from_snapshot)
    print(json.dumps(dict(collection_status=analysis['collection_status'], all_terminal=analysis['all_submitted_terminal'], all_usable=analysis['all_usable'],
                         accounting_unknown=analysis['accounting_unknown_job_ids'],
                         guard_plan=analysis['guard_plan'], formal_training_released=False), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
