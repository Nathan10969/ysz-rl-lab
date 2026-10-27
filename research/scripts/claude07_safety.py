"""Local/runtime safety helpers for Claude07 (cluster Python >=3.6, stdlib only).

Scientific dependencies and resource lanes are deliberately separate. A failed
task never writes a pool stop. Only an unsafe spending projection may do that.
"""
import argparse
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

TERMINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'BLOCKED'}
POOL_STOP = 'CLAUDE07_POOL_STOP.json'


def _nonnegative(value, label):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError('Invalid nonnegative number: '+label)
    return number


def budget_projection(jobs, observed, historical):
    names = [j['name'] for j in jobs]
    if len(set(names)) != len(names) or set(observed)-set(names):
        raise ValueError('Duplicate jobs or unrecognized observations; reconcile first')
    actual = _nonnegative(historical, 'historical')
    remaining = 0.0
    for job in jobs:
        row = observed.get(job['name'], {})
        spent = _nonnegative(row.get('actual', 0), job['name']+' actual')
        cap = _nonnegative(job['cap'], job['name']+' cap')
        actual += spent
        state = row.get('state', 'UNKNOWN').split()[0].rstrip('+')
        if state not in TERMINAL:
            remaining += max(0, cap-spent)
    return dict(actual=actual, remaining_caps=remaining,
                actual_plus_remaining_caps=actual+remaining)


def budget_status(jobs, observed, historical, pool_limit, round_limit):
    projection = budget_projection(jobs, observed, historical)
    current = budget_projection(jobs, observed, 0)
    reasons = []
    if projection['actual_plus_remaining_caps'] > float(pool_limit)+1e-9:
        reasons.append('existing_pool_cap_exceeded')
    if current['actual_plus_remaining_caps'] > float(round_limit)+1e-9:
        reasons.append('round_cap_exceeded')
    for job in jobs:
        if float(observed.get(job['name'], {}).get('actual', 0)) > float(job['cap'])+1e-9:
            reasons.append('job_cap_exceeded:'+job['name'])
    return dict(projection, safe=not reasons, reasons=reasons,
                round_projection=current['actual_plus_remaining_caps'],
                pool_limit=pool_limit, round_limit=round_limit)


def blocked_jobs(jobs, failed_names):
    known = {j['name'] for j in jobs}
    if any(set(j.get('depends_on', []))-known for j in jobs):
        raise ValueError('Unknown scientific dependency')
    blocked = set(failed_names)
    while True:
        expanded = blocked | {j['name'] for j in jobs if set(j.get('depends_on', [])) & blocked}
        if expanded == blocked:
            break
        blocked = expanded
    return [j['name'] for j in jobs if j['name'] in blocked-set(failed_names)]


def dependency_flags(job, submitted_ids):
    parts = []
    for key, mode in [('resource_afterany', 'afterany'), ('depends_on', 'afterok')]:
        deps = job.get(key, [])
        if deps:
            ids = [str(submitted_ids[name]) for name in deps]
            if not all(i.isdigit() for i in ids):
                raise ValueError('Invalid Slurm dependency id')
            parts.append(mode+':'+':'.join(ids))
    return ['--dependency='+','.join(parts)] if parts else []


def last_complete_calculation(xml_text, expected_index=None):
    """Extract the final closed calculation, never CONTCAR or a partial step."""
    last = None
    count = 0
    incomplete = False
    symbols = []
    try:
        for _, node in ET.iterparse(io.StringIO(xml_text), events=('end',)):
            if node.tag == 'atominfo':
                symbols = [rc.find('c').text.strip() for rc in node.findall('array[@name="atoms"]/set/rc')]
            if node.tag != 'calculation':
                continue
            count += 1
            def vectors(path):
                found = node.find(path)
                if found is None:
                    raise ValueError('Complete calculation lacks '+path)
                rows = [[float(x) for x in v.text.split()] for v in found.findall('v')]
                if not rows or any(len(v) != 3 or not all(math.isfinite(x) for x in v) for v in rows):
                    raise ValueError('Invalid calculation vectors: '+path)
                return rows
            last = dict(basis=vectors('structure/crystal/varray[@name="basis"]'),
                        positions=vectors('structure/varray[@name="positions"]'),
                        forces=vectors('varray[@name="forces"]'),
                        scf_steps=len(node.findall('scstep')),
                        calculation_xml=ET.tostring(node, encoding='unicode'))
            if len(last['basis']) != 3 or len(last['positions']) != len(last['forces']) or last['scf_steps'] < 1:
                raise ValueError('Incomplete ionic structure or forces')
            node.clear()
    except ET.ParseError:
        incomplete = True
    if last is None or (expected_index is not None and count != expected_index):
        raise ValueError('Expected complete calculation %s; found %s' % (expected_index, count))
    last.update(completed_calculations=count, frame_1based=count,
                trailing_xml_incomplete=incomplete, symbols=symbols)
    return last


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=str(path.parent),
                                         prefix='.'+path.name+'.', suffix='.tmp', delete=False) as handle:
            temporary = handle.name
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def record_task_status(directory, state, reason, exit_code=None):
    if state not in {'RUNNING', 'COMPLETED', 'FAILED', 'BLOCKED'}:
        raise ValueError('Invalid task state')
    directory = Path(directory)
    data = dict(name=directory.name, state=state, reason=reason, exit_code=exit_code,
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    atomic_json(directory/'TASK_STATUS.json', data)
    return data


def write_budget_stop(pool, status):
    if status['safe'] or not status['reasons']:
        raise ValueError('Pool stop is reserved for a demonstrated budget risk')
    atomic_json(Path(pool)/POOL_STOP, dict(status, scope='this_resource_pool_only'))


def verify_hash_manifest(directory):
    directory = Path(directory)
    seen = set()
    for line in (directory/'REVIEWED_INPUTS.sha256').read_text().splitlines():
        try:
            digest, filename = line.split('  ', 1)
        except ValueError:
            raise ValueError('Malformed input hash manifest')
        if (len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or filename in seen or Path(filename).name != filename
                or filename in {'.', '..'} or (directory/filename).is_symlink()):
            raise ValueError('Unsafe or duplicate hash manifest entry')
        seen.add(filename)
        path = directory/filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Reviewed input hash mismatch: '+filename)
    if not seen:
        raise ValueError('Empty reviewed input manifest')
    return seen


def validate_allocation(meta, environ):
    expected_threads = meta['cores'] if meta['gpus'] else 1
    for key, expected in [('SLURM_NTASKS', meta['mpi_ranks']),
                          ('SLURM_CPUS_PER_TASK', expected_threads)]:
        if str(environ.get(key, '')) != str(expected):
            raise ValueError('Allocation fingerprint mismatch: '+key)


def validate_output(xml_text, outcar_text, is_relax):
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        return dict(valid=False, error='incomplete_or_invalid_XML: '+str(error))
    node = root.find('.//parameters//i[@name="NELM"]')
    nelm = int(node.text) if node is not None else 120
    calculations = root.findall('calculation')
    scf = [len(c.findall('scstep')) for c in calculations]
    reached = outcar_text.count('aborting loop because EDIFF is reached')
    electronic = bool(scf) and all(0 < n < nelm for n in scf) and reached >= len(scf)
    ionic = 'reached required accuracy - stopping structural energy minimisation' in outcar_text
    return dict(valid=electronic and (not is_relax or ionic),
                electronic_converged=electronic, ionic_converged=ionic if is_relax else None,
                n_ionic_outputs=len(scf), scf_steps=scf, nelm=nelm, EDIFF_markers=reached)


def claim_task(directory):
    directory = Path(directory)
    prior_outputs = ['TASK_STATUS.json', 'OUTCAR', 'CONTCAR', 'vasprun.xml', 'OSZICAR',
                     'log.txt', 'POTCAR', 'RESULT_CHECK.json', 'runtime_input_hashes.sha256',
                     'RUNTIME_FINGERPRINT.json', 'gpu_samples.csv', 'cuda_preflight_filtered.json',
                     'cuda_preflight_preload.json']
    if any((directory/name).exists() for name in prior_outputs):
        raise FileExistsError('Existing task state/output; no automatic retry or overwrite')
    # O_EXCL makes concurrent/repeated starts fail without replacing any result.
    with (directory/'RUN_CLAIM.json').open('x') as handle:
        json.dump(dict(slurm_job_id=os.environ.get('SLURM_JOB_ID')), handle)
    try:
        verify_hash_manifest(directory)
        meta = json.loads((directory/'job.json').read_text())
        if (directory.parent/POOL_STOP).exists():
            raise ValueError('This resource pool has a budget stop')
        for name in meta['depends_on']:
            status = directory.parent/name/'TASK_STATUS.json'
            if not status.exists() or json.loads(status.read_text())['state'] != 'COMPLETED':
                record_task_status(directory, 'BLOCKED', 'Scientific dependency unavailable: '+name, 91)
                return 91
        validate_allocation(meta, os.environ)
        record_task_status(directory, 'RUNNING', 'Reviewed hashes and allocation verified')
        return 0
    except Exception as error:
        record_task_status(directory, 'FAILED', 'preflight: '+str(error), 92)
        return 92


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['claim', 'finish', 'check-result'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('exit_code', type=int, nargs='?')
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.action == 'claim':
        raise SystemExit(claim_task(directory))
    if args.action == 'finish':
        if args.exit_code is None:
            parser.error('finish requires exit_code')
        record_task_status(directory, 'COMPLETED' if args.exit_code == 0 else 'FAILED',
                           'VASP and convergence checks completed' if args.exit_code == 0
                           else 'Task failed; independent tasks remain eligible; no automatic retry',
                           args.exit_code)
        return
    meta = json.loads((directory/'job.json').read_text())
    try:
        result = validate_output((directory/'vasprun.xml').read_text(errors='replace'),
                                 (directory/'OUTCAR').read_text(errors='replace'), meta['kind'] == 'relax')
    except Exception as error:
        result = dict(valid=False, error=str(error))
    atomic_json(directory/'RESULT_CHECK.json', result)
    print(json.dumps(result))
    raise SystemExit(0 if result['valid'] else 93)


if __name__ == '__main__':
    main()
