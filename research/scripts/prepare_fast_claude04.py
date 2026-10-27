"""Prepare only the two explicitly approved Claude04 fast comparisons, without submission."""
import datetime, hashlib, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / 'calibration'
REFERENCE = 'REVIEW_CLAUDE_04_L00.md'
GAM_SHA = '221ea3a02906c0eb88dcf504f5d2f9c1788a128264a8254b06819ae8c1a6ad1e'
STD_SHA = 'f51016d290697d9c63e27feb71a278f7127309f6e23696cd6d68c3f1c43a4159'
REMOTE = '/online1/ycsc_hzw/hd100371/Jony/runs/01_calib'

def parameters(text):
    return {line.split('=', 1)[0].strip(): line.split('=', 1)[1].strip()
            for line in text.splitlines() if '=' in line and not line.lstrip().startswith('#')}

def main():
    assert not (C / 'FAST_SUBMISSIONS.json').exists(), 'Already submitted: reconcile instead of preparing again'
    assert not (C / 'execution_claude04').exists(), 'Prepared directory already exists: inspect rather than overwrite'
    archive = ROOT / 'archive' / 'Claude04_before_execution'
    archive.mkdir(exist_ok=False)
    for name in ['PLAN.md', 'P0_RESULTS.md']:
        if (ROOT / name).exists(): shutil.copy2(ROOT / name, archive / name)
    for name in ['BUDGET.json', 'REVIEW_STATUS.json', 'P0_SUBMISSIONS.json', 'P0_ANALYSIS.json', 'P1_GATE.json']:
        if (C / name).exists(): shutil.copy2(C / name, archive / name)
    prior = json.loads((C / 'CLAUDE03_JOB_LIST.json').read_text())
    jobs = []
    for name, old_name, config in [('05f_L00', '03q_L00_gamma64', 'L00'),
                                   ('06f_L01', '04q_L01_gamma64', 'L01')]:
        original = next(j for j in prior if j['name'] == old_name)
        src = C / 'execution_claude03' / old_name
        d = C / 'execution_claude04' / name
        d.mkdir(parents=True)
        for filename, digest in original['input_hashes'].items():
            assert hashlib.sha256((src / filename).read_bytes()).hexdigest() == digest
        for filename in ['POSCAR', 'KPOINTS', 'atom_mapping.json']:
            shutil.copy2(src / filename, d / filename)
        assert (d / 'KPOINTS').read_text().splitlines()[3].split() == ['1', '1', '1']
        lines = (src / 'INCAR').read_text().splitlines()
        lines = [dict(EDIFF='EDIFF = 1E-5', LREAL='LREAL = Auto').get(line.split('=')[0].strip(), line)
                 for line in lines]
        (d / 'INCAR').write_text('\n'.join(lines) + '\nAMIX = 0.2\nBMIX = 0.0001\n')
        before, after = parameters((src / 'INCAR').read_text()), parameters((d / 'INCAR').read_text())
        assert {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)} == {'EDIFF', 'LREAL', 'AMIX', 'BMIX'}
        sub = (src / 'sub.sh').read_text()
        sub = sub.replace('# Claude03: remaining P0 approved; P1 conditional on converged 03/04 each <=40 core-hours. P2+ held.',
                          '# Claude04: two fast single points only. P1 and all later phases remain held.')
        sub = sub.replace(old_name, name).replace('--ntasks=64', '--ntasks=128').replace('--mem=96000M', '--mem=192000M')
        sub = sub.replace('/vasp_std"', '/vasp_gam"').replace(STD_SHA, GAM_SHA)
        sub = sub.replace('set -euo pipefail\n', 'set -euo pipefail\nsha256sum -c REVIEWED_INPUTS.sha256\n')
        (d / 'sub.sh').write_text(sub)
        (d / 'sub.sh').chmod(0o755)
        reviewed = {f: hashlib.sha256((d / f).read_bytes()).hexdigest()
                    for f in ['INCAR', 'KPOINTS', 'POSCAR', 'atom_mapping.json']}
        (d / 'REVIEWED_INPUTS.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in reviewed.items()))
        job = dict(original, name=name, id='fast_' + config, logical_job='fast_' + config,
                   phase='CLAUDE04_fast', cores=128, limit_hours=1, hard_core_hours=128,
                   lreal='Auto', ediff=1e-5, amix=.2, bmix=.0001, binary='vasp_gam',
                   binary_sha256=GAM_SHA, approval_reference=REFERENCE,
                   review_status='APPROVED_CLAUDE04', remote_directory=REMOTE + '/' + name,
                   physical_source_attempt=old_name)
        job['input_hashes'] = {f: hashlib.sha256((d / f).read_bytes()).hexdigest()
                               for f in ['INCAR', 'KPOINTS', 'POSCAR', 'sub.sh', 'atom_mapping.json', 'REVIEWED_INPUTS.sha256']}
        (d / 'job.json').write_text(json.dumps(job, indent=2) + '\n')
        jobs.append(job)
    (C / 'CLAUDE04_JOB_LIST.json').write_text(json.dumps(jobs, indent=2) + '\n')
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    budget = json.loads((C / 'BUDGET.json').read_text())
    budget.update(approval_reference=REFERENCE, fast_hard_core_hours=256,
                  original_remaining_p2plus_core_hour_caps=2496,
                  fast_reallocated_from_unreleased_p2plus_core_hours=256,
                  remaining_p2plus_core_hour_caps=2240,
                  budget_reallocation_note='Claude04 fast256 replaces256 of the unsubmitted, unreleased P2+ reserve; future P2+ list must be replanned before release.',
                  fast_actual_core_hours=0, remaining_fast_core_hour_caps=256)
    (C / 'BUDGET.json').write_text(json.dumps(budget, indent=2) + '\n')
    status = json.loads((C / 'REVIEW_STATUS.json').read_text())
    status.update(approval_reference=REFERENCE, claude_review='CLAUDE04_APPROVED_TWO_FAST_SINGLE_POINTS',
                  p0_state='CLAUDE04_FAST_PREPARED', fast_jobs=[j['name'] for j in jobs],
                  fast_hard_core_hours=256, old_p0_hard_core_hours=216,
                  later_phases='P1_AND_LATER_HELD_CLAUDE04', p1_state='HELD_CLAUDE04',
                  p1_gate={'allowed': False, 'reason': 'Claude04 supersedes the old conditional P1 release'},
                  p2_and_later='HELD', further_submissions_allowed=True,
                  further_submissions_scope='Exactly05f_L00 and06f_L01; no retries, P1, training, relaxation or GPU jobs',
                  old_L01_disposition='Leave1882470 to natural completion or existing1hour timeout; do not rerun old protocol',
                  awaiting='Fast single-point results for Claude review', updated_at_utc=now)
    status.pop('halt_reason', None)
    (C / 'REVIEW_STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
    (C / 'P1_GATE.json').write_text(json.dumps({'allowed': False, 'approval_reference': REFERENCE,
        'reason': 'Explicit P1 hold; the Claude03 <=40 core-hour gate is superseded', 'checked_at_utc': now}, indent=2) + '\n')
    print(json.dumps({'prepared': [j['name'] for j in jobs], 'hard_core_hours': 256,
                      'remaining_unreleased_P2plus_reserve': 2240, 'submitted': False}, indent=2))

if __name__ == '__main__': main()
