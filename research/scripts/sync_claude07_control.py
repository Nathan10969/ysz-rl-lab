"""Synchronize Review07 summaries into new remote control/ directories.

This never touches runtime STOP/TASK_STATUS markers, submission ledgers, old
calibration roots, or executable inputs. Without --apply it only lists files.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from collect_claude07 import POOLS, ROOT
from submit_claude05 import ssh

SYNC_POOLS = dict(POOLS, TRAINING_SMOKE=dict(home='training_smoke', host='your-cluster',
    remote='/CLUSTER_HOME/R_YSZ01_dopant_vacancy_agent/training_smoke'))


def control_files(root, home):
    files = [home/name for name in ['JOB_LIST.json', 'JOB_MANIFEST.json', 'PREPARED_LEDGER.json', 'BUDGET.json',
                                    'ANALYSIS.json', 'SCIENCE.json', 'STATUS.json', 'SCF_TIMINGS.csv']]
    files += [root/name for name in ['REVIEW_CLAUDE_07_gates.md', 'CLAUDE07_RESULTS.md',
                                     'TRAINING_PROBE_PLAN.md', 'TRAINING_ENV_SPEC.md', 'CLAUDE07_PREFLIGHT.md',
                                     'REVIEW_CLAUDE_08_RL_arm.md', 'CLAUDE08_EXECUTION.md',
                                     'CLAUDE08_FOLLOWUP.md', 'BUDGET_REPLAN.md']]
    files.append(root/'training_smoke/REVIEW08_AUTHORIZATION.json')
    files.append(root/'training_smoke/SCHEDULE_AMENDMENTS.json')
    files.append(root/'training_smoke/SINGLE_GPU_PRIORITY_AMENDMENT.json')
    files.append(root/'training_smoke/WORKLOAD_AMENDMENTS.json')
    files.append(root/'calibration/REVIEW_STATUS.json')
    science = root/'calibration/claude07/SCIENCE.json'
    if science not in files:
        files.append(science)
    return [path for path in files if path.is_file()]


def synchronize(root, name, settings, apply=False):
    files = control_files(root, root/settings['home'])
    if not files:
        raise ValueError('No Review07 control summaries found')
    payload = {path.name: dict(data=base64.b64encode(path.read_bytes()).decode('ascii'),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in files}
    summary = dict(pool=name, host=settings['host'], remote_directory=settings['remote']+'/control',
                   files={name: value['sha256'] for name, value in payload.items()}, applied=False)
    if not apply:
        return summary
    code = '''import base64,hashlib,json,os,tempfile
from pathlib import Path
root=Path(REMOTE)/'control'
root.mkdir(parents=True,exist_ok=True)
payload=PAYLOAD
for name,item in payload.items():
    if Path(name).name!=name or name in ['TASK_STATUS.json','CLAUDE07_POOL_STOP.json','SUBMISSIONS.json']:
        raise ValueError('Unsafe control summary name')
    data=base64.b64decode(item['data'])
    if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Transfer hash mismatch')
    with tempfile.NamedTemporaryFile(mode='wb',dir=str(root),prefix='.'+name+'.',delete=False) as handle:
        handle.write(data);handle.flush();os.fsync(handle.fileno());temporary=handle.name
    os.replace(temporary,str(root/name))
print(json.dumps({'verified_files':list(payload),'runtime_markers_untouched':True}))
'''.replace('REMOTE', repr(settings['remote'])).replace('PAYLOAD', repr(payload))
    result = ssh(settings['host'], 'python3 -', code.encode(), timeout=120)
    summary.update(applied=True, remote_result=json.loads(result.stdout))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--pools', nargs='+', choices=sorted(SYNC_POOLS), default=list(SYNC_POOLS))
    parser.add_argument('--apply', action='store_true', help='Perform atomic remote control-summary writes')
    args = parser.parse_args(argv)
    print(json.dumps([synchronize(args.root, name, SYNC_POOLS[name], args.apply) for name in args.pools], indent=2))


if __name__ == '__main__':
    main()
