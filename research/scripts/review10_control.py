"""Read-only remote recovery and parent-only accounting for the Review10 package.

Never submits, prepares, cancels or retries a GPU job. Frozen four-parent history
and the three new stage caps share the original10 card-hour pool.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

import collect_claude07_smoke as collect
from claude07_safety import TERMINAL, atomic_json

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / 'training_smoke'
PACKAGE = HOME / 'claude10'
STAGES = ('teacher', 'sft', 'grpo')


def aggregate_budget(baseline, authorization, groups):
    """Unknown parent bills keep the unspent cap, including terminal-looking rows."""
    if authorization['package_cap_card_hours'] != 6:
        raise ValueError('Review10 package cap must stay6 inside original10')
    stages = authorization['stages']
    if len(stages) != 3 or {s['stage'] for s in stages} != set(STAGES):
        raise ValueError('Exactly teacher/SFT/GRPO stage caps are authorized')
    if any(s['hard_card_hours'] != 2 for s in stages):
        raise ValueError('Registered stage caps are two card-hours each')
    rows = copy.deepcopy(baseline['jobs'])
    identities = [str(j['slurm_job_id']) for j in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate baseline parent')
    ids = set(identities)
    complete = baseline.get('accounting_complete') is True
    unknown = list(baseline.get('accounting_unknown_job_ids', []))
    actual = reserve = 0.
    new_rows = []
    for stage in stages:
        group = groups[stage['stage']]
        jobs = group.get('jobs', [])
        if len(jobs) > 1:
            raise ValueError('No stage may submit twice')
        stage_unknown = list(group.get('accounting_unknown_job_ids', []))
        unknown.extend(stage_unknown)
        known = group.get('accounting_complete') is True and not stage_unknown
        complete = complete and known
        budget_ids = {str(j.get('slurm_job_id', '')) for j in jobs}
        submitted_ids = {str(x) for x in group.get('submitted_job_ids', budget_ids)}
        if submitted_ids != budget_ids:
            raise ValueError('Submission parent missing from stage budget; review raw evidence before publication')
        pending_intent = bool(group.get('pending_submission'))
        if pending_intent and jobs:
            raise ValueError('Submission intent coexists with a parent; reconcile without submitting again')
        if pending_intent:
            complete = False
            unknown.append('UNRESOLVED_SUBMISSION_' + stage['stage'])
        withheld = stage.get('state') == 'NOT_RELEASED_UPSTREAM_FAILURE'
        if withheld and (jobs or submitted_ids or pending_intent or not known or stage_unknown):
            raise ValueError('Cannot withhold a submitted or uncertain stage')
        if not jobs:
            reserve += 0 if withheld else 2
            continue
        job = jobs[0]
        sid = str(job.get('slurm_job_id', ''))
        if not sid.isdigit() or sid in ids:
            raise ValueError('Missing or duplicate new parent ID')
        ids.add(sid)
        amount = job.get('actual_card_hours', 0)
        if (isinstance(amount, bool) or not isinstance(amount, (int, float))
                or not math.isfinite(amount) or not 0 <= amount <= 2 + 1e-6):
            raise ValueError('Invalid stage cost or cap overrun')
        if (job.get('hard_gpu_hours') != 2 or job.get('gpus') != 1
                or job.get('allocation_gpu_count', 1) != 1
                or job.get('unexpected_gpu_allocation') or job.get('hard_cap_exceeded')):
            raise ValueError('Stage resource/allocation anomaly')
        actual += amount
        if not known or job.get('state') not in TERMINAL:
            reserve += max(0, 2 - amount)
        if not known:
            unknown.append(sid)
        new_rows.append(copy.deepcopy(job))
    if actual + reserve > 6 + 1e-9:
        raise ValueError('Review10 package cap exceeded')
    total = baseline['actual_card_hours'] + actual
    if not math.isfinite(total) or total < 0 or total + reserve > 10 + 1e-9:
        raise ValueError('Original smoke10 pool cap exceeded')
    package = dict(schema='review10-budget-v1', authorized_card_hours=6,
        actual_card_hours=actual, remaining_caps=reserve,
        actual_plus_remaining_caps=actual + reserve, jobs=new_rows,
        accounting_complete=complete, accounting_unknown_job_ids=sorted(set(unknown)),
        updated_at_utc=collect.now(), stage_reservations=copy.deepcopy(stages))
    result = copy.deepcopy(baseline)
    result.update(jobs=rows + new_rows, actual_card_hours=total, actual_gpu_hours=total,
        remaining_caps=reserve, actual_plus_remaining_caps=total + reserve,
        unallocated_gpu_hours=10 - total - reserve, accounting_complete=complete,
        accounting_unknown_job_ids=sorted(set(unknown)),
        review10_package_actual_card_hours=actual, review10_package_cap_card_hours=6,
        review10_reservation_card_hours=reserve, review10_authorization='claude10/AUTHORIZATION.json',
        conditional_GRPO_reservation_card_hours=sum(
            2 for s in stages if s['stage'] == 'grpo'
            and not groups['grpo'].get('jobs') and s.get('state') != 'NOT_RELEASED_UPSTREAM_FAILURE'),
        active_approval_reference='REVIEW_CLAUDE_10_v2_sft.md plus direct user approval',
        accounting_updated_at_utc=collect.now(), formal_training_released=False,
        billing='Each verified Slurm allocation parent once; actual AllocTRES GPUs times ElapsedRaw/3600',
        submission_rule='Review10 serial teacher -> verified newv2 SFT76/80 -> unique GRPO19/20; no automatic retry')
    return result, package


def recover_stage(stage):
    home = PACKAGE / stage
    ledger = collect.load(home / 'SUBMISSIONS.json')
    if not ledger or not ledger.get('jobs'):
        return
    manifest = collect.load(home / 'JOB_MANIFEST.json')
    if not manifest or len(ledger['jobs']) != 1:
        raise ValueError('Expected exactly one frozen stage manifest and parent')
    previous = collect.load(home / 'BUDGET.json')
    try:
        snapshot = collect.query(home, manifest, 'your-cluster')
        if snapshot['account']['code'] != 0:
            raise RuntimeError('sacct query returned failure; raw evidence in LATEST.json')
        records, risks, unknown = collect.parse_accounting(snapshot['account']['stdout'], manifest,
            ledger, previous, snapshot.get('allocation_proofs'))
        if risks:
            atomic_json(home / 'ANALYSIS.json', dict(collection_status='DEFERRED', risks=risks,
                evidence='LATEST.json', checked_at_utc=collect.now()))
            raise ValueError('Resource/allocation anomaly requires raw evidence review')
        collect.reconcile_budget(home, ledger, records, unknown, str(home / 'LATEST.json'))
        jobs = []
        for job in ledger['jobs']:
            sid = str(job['slurm_job_id'])
            account = records.get(sid, {})
            terminal = account.get('state') in TERMINAL
            out = home / 'outputs' / (stage + '_' + sid)
            retrieved = False
            result = None
            if terminal:
                out.parent.mkdir(parents=True, exist_ok=True)
                # Raw output directories are copied only after terminal observation.
                for remote in (job['remote_output'], job['remote_log']):
                    destination = out.parent if remote == job['remote_output'] else out
                    subprocess.run(['scp', '-r', 'your-cluster:' + remote, str(destination)],
                        check=True, capture_output=True, timeout=120)
                result = collect.load(out / 'result.json')
                retrieved = out.is_dir() and (out / Path(job['remote_log']).name).is_file()
                atomic_json(out / 'RETRIEVED_TERMINAL.json',
                    dict(at_utc=collect.now(), state=account.get('state'), retrieved=retrieved))
            jobs.append(dict(job, state=account.get('state', 'ACCOUNTING_UNKNOWN'),
                accounting=account, terminal=terminal, terminal_outputs_retrieved=retrieved,
                result=collect.json_evidence(result), independently_verified=False))
        atomic_json(home / 'ANALYSIS.json', dict(checked_at_utc=collect.now(),
            collection_status='DEFERRED' if unknown else 'COLLECTED',
            accounting_unknown_job_ids=unknown, risks=[], jobs=jobs,
            all_terminal_recovered=all(j['terminal'] and j['terminal_outputs_retrieved'] for j in jobs)))
    except Exception as error:
        # A failed query/copy never proves purge or a free reservation.
        budget = collect.load(home / 'BUDGET.json', previous)
        budget.update(accounting_complete=False,
            accounting_unknown_job_ids=[str(j['slurm_job_id']) for j in ledger['jobs']],
            collection_error=type(error).__name__ + ': ' + str(error))
        atomic_json(home / 'BUDGET.json', budget)
        failure = dict(collection_status='DEFERRED', checked_at_utc=collect.now(),
            error=type(error).__name__ + ': ' + str(error), raw_evidence='LATEST.json')
        atomic_json(home / 'COLLECTION_FAILURE.json', failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true', required=True)
    parser.add_argument('--stage', choices=STAGES)
    args = parser.parse_args()
    if (PACKAGE / 'recovery_v21' / 'AUTHORIZATION.json').is_file():
        raise RuntimeError('recovery_v21 is current; the old collector cannot overwrite its reservations')
    authorization = collect.load(PACKAGE / 'AUTHORIZATION.json')
    for filename, expected in authorization['immutable_submission_sha256'].items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != expected:
            raise ValueError('Historical submission ledger changed: ' + filename)
    for stage in ((args.stage,) if args.stage else STAGES):
        recover_stage(stage)
    groups = {}
    for stage in STAGES:
        group = collect.load(PACKAGE / stage / 'BUDGET.json')
        ledger = collect.load(PACKAGE / stage / 'SUBMISSIONS.json')
        if (not isinstance(ledger, dict) or not isinstance(ledger.get('jobs'), list)
                or len(ledger['jobs']) > 1):
            raise ValueError('Missing or malformed stage submission ledger')
        group['submitted_job_ids'] = [str(j['slurm_job_id']) for j in ledger['jobs']]
        group['pending_submission'] = bool(ledger.get('pending_submission'))
        groups[stage] = group
    root, package = aggregate_budget(collect.load(PACKAGE / 'BASELINE_BUDGET.json'), authorization, groups)
    atomic_json(PACKAGE / 'BUDGET.json', package)
    atomic_json(HOME / 'BUDGET.json', root)
    analysis = collect.load(HOME / 'ANALYSIS.json')
    analysis.update(review10_stages={s: collect.load(PACKAGE / s / 'ANALYSIS.json', {}) for s in STAGES},
                    review10_updated_at_utc=collect.now())
    atomic_json(HOME / 'ANALYSIS.json', analysis)
    print(json.dumps({key: root[key] for key in ('actual_card_hours', 'remaining_caps',
        'actual_plus_remaining_caps', 'unallocated_gpu_hours', 'accounting_complete')}))


if __name__ == '__main__':
    main()
