"""Recover approved v22 stages and account the original10 GPU pool.

This controller never submits, prepares, cancels, requeues, retries, or runs a
model. --publish uses only saved local evidence. --collect makes one read-only
query per submitted stage and reuses the audited Slurm/allocation parser.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import collect_claude07_smoke as collect
from claude07_safety import TERMINAL, atomic_json

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / 'training_smoke'
PACKAGE = HOME / 'claude10/recovery_v22'
CAPS = {'teacher': 2., 'sft': 1.85, 'grpo': 2.}
STAGES = tuple(CAPS)
FROZEN_SECONDS = {'633': 35, '635': 9, '648': 36, '650': 72, '652': 507, '656': 153}
HISTORY_SHA256 = '63bf14929cb6641dd48465603fbf726b52d9935aa31aa7f56f71d1a0ce4a1df7'
BASELINE_SHA256 = '7ff5adde926387658519eac73e86fb90cea7ba0814162ecdf18c8cad416c9ac6'


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative numeric value required: ' + label)
    return float(value)


def _verify_allocated_tres(text):
    """Check raw observed resources; never rewrite the stored bill or columns."""
    if not isinstance(text, str):
        raise ValueError('Saved allocation TRES required')
    entries = [field.split('=', 1) for field in text.split(',') if '=' in field]
    allocation = dict(entries)
    if len(entries) != len(allocation):
        raise ValueError('Duplicate allocation resource fields')
    gpu_values = [v for k, v in allocation.items() if k == 'gres/gpu' or k.startswith('gres/gpu:')]
    if allocation.get('cpu') != '6' or not gpu_values or any(value != '1' for value in gpu_values):
        raise ValueError('Observed allocation must be1GPU/6CPU')
    memory = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)([KMGTP]?)', allocation.get('mem', ''))
    if memory is None:
        raise ValueError('Complete observed memory allocation required')
    gib = float(memory[1]) * 1024. ** ({'':-3, 'K':-2, 'M':-1, 'G':0, 'T':1, 'P':2}[memory[2]])
    if not math.isfinite(gib) or abs(gib-90) > 1e-10:
        raise ValueError('Observed allocation must be90GiB')


def validate_authorization(auth):
    if not isinstance(auth, dict) or auth.get('schema') != 'review10-v22-authorization-v1':
        raise ValueError('Missing/malformed recovery_v22 authorization')
    if (not isinstance(auth.get('state'), str)
            or not re.fullmatch(r'[A-Z][A-Z0-9_]{0,95}', auth['state'])):
        raise ValueError('Missing/malformed recovery_v22 authorization state')
    required = dict(root_pool_card_hours=10,
                    max_concurrent_GPU_jobs=1, max_new_teacher_submissions=1)
    if _number(auth.get('package_total_cap_including652_656_card_hours'), 'package cap') != 6.5:
        raise ValueError('Package cap must include652/656 inside6.5')
    if any(type(auth.get(key)) is not int or auth[key] != value for key, value in required.items()):
        raise ValueError('Recovery authorization caps or concurrency changed')
    if (auth.get('direct_user_confirmation') is not True or auth.get('automatic_retry') is not False
            or auth.get('formal_training_allowed') is not False or auth.get('new_DFT_allowed') is not False):
        raise ValueError('Direct bounded recovery authorization required; formal work remains disabled')
    caps = auth.get('stage_caps_card_hours')
    if not isinstance(caps, dict) or set(caps) != set(CAPS):
        raise ValueError('Exactly teacher/SFT/GRPO caps required')
    if any(_number(caps[s], s+' cap') != CAPS[s] for s in STAGES):
        raise ValueError('Registered recovery caps must stay teacher2/SFT1.85/GRPO2')
    if (abs(_number(auth.get('historical_package_actual_card_hours'), '652/656 cost')-660/3600) > 1e-12
            or abs(_number(auth.get('historical_root_actual_card_hours'), 'old6 cost')-812/3600) > 1e-12):
        raise ValueError('Previously billed parent costs cannot be reset')


def validate_baseline(baseline):
    if (not isinstance(baseline, dict) or baseline.get('accounting_complete') is not True
            or baseline.get('accounting_unknown_job_ids') != [] or baseline.get('remaining_caps') != 0
            or baseline.get('authorized_card_hours') != 10):
        raise ValueError('Frozen six-parent baseline must have closed accounting')
    rows = baseline.get('jobs')
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError('All six previously billed parents must remain in baseline')
    seen = set()
    for job in rows:
        sid = str(job.get('slurm_job_id', ''))
        if sid not in FROZEN_SECONDS or sid in seen:
            raise ValueError('Missing or duplicate historical parent')
        seen.add(sid)
        amount = _number(job.get('actual_card_hours'), 'baseline parent fee')
        if (abs(amount-FROZEN_SECONDS[sid]/3600) > 1e-12 or job.get('state') not in TERMINAL
                or job.get('gpus') != 1 or job.get('allocation_gpu_count') != 1
                or job.get('unexpected_gpu_allocation') or job.get('hard_cap_exceeded')):
            raise ValueError('Frozen parent fee, terminal state, or allocation changed')
        row = job.get('sacct', {})
        if (str(row.get('job_id', '')) != sid or row.get('elapsed_seconds') != FROZEN_SECONDS[sid]
                or row.get('state') != job['state']):
            raise ValueError('Frozen parent raw accounting binding changed')
        _verify_allocated_tres(row.get('alloc_tres'))
    if abs(_number(baseline.get('actual_card_hours'), 'baseline total')-812/3600) > 1e-12:
        raise ValueError('Frozen baseline total does not match all six parents')


def validate_frozen_history():
    """No metadata publication or remote reads before immutable-ledger checks."""
    manifest_path = PACKAGE/'HISTORY_IMMUTABLE_SHA256.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != HISTORY_SHA256:
        raise ValueError('Immutable328-file history manifest bytes changed')
    manifest = collect.load(manifest_path)
    expected = manifest.get('files') if isinstance(manifest, dict) else None
    if (not isinstance(expected, dict) or manifest.get('count') != 328 or len(expected) != 328):
        raise ValueError('Exactly328 frozen historical file hashes required')
    for filename, digest in expected.items():
        name = Path(filename)
        if (not isinstance(filename, str) or name.is_absolute() or '..' in name.parts
                or name.as_posix() != filename or not re.fullmatch(r'[0-9a-f]{64}', str(digest))):
            raise ValueError('Malformed immutable history path/hash')
        path = ROOT/name
        if (path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError('Historical file changed: ' + filename)
    if hashlib.sha256((PACKAGE/'BASELINE_BUDGET.json').read_bytes()).hexdigest() != BASELINE_SHA256:
        raise ValueError('Frozen six-parent baseline bytes changed')


def aggregate_budget(baseline, authorization, groups):
    """Preserve unknown-fee reservations; include652/656 once inside additional6.5."""
    validate_authorization(authorization)
    validate_baseline(baseline)
    if not isinstance(groups, dict) or set(groups) != set(STAGES):
        raise ValueError('Exactly three recovery stage budgets required')
    ids = set(FROZEN_SECONDS)
    actual = reserve = 0.
    unknown, new_rows, states = [], [], {}
    for stage, cap in CAPS.items():
        group = groups[stage]
        if not isinstance(group, dict) or _number(group.get('authorized_card_hours'), stage+' authorization') != cap:
            raise ValueError('Stage budget authorization differs from fixed cap')
        jobs, submitted = group.get('jobs'), group.get('submitted_job_ids')
        if (not isinstance(jobs, list) or len(jobs) > 1 or not isinstance(submitted, list)
                or len(submitted) > 1 or not all(isinstance(j, dict) for j in jobs)):
            raise ValueError('At most one parent and an explicit ledger binding required per stage')
        submitted_ids = {str(sid) for sid in submitted}
        budget_ids = {str(job.get('slurm_job_id', '')) for job in jobs}
        if submitted_ids != budget_ids:
            raise ValueError('Stage submission ledger/budget parent mismatch')
        if type(group.get('pending_submission', False)) is not bool:
            raise ValueError('Explicit Boolean pending-submission binding required')
        pending = group.get('pending_submission', False)
        if pending and jobs:
            raise ValueError('Parent and another submission intent cannot coexist')
        stage_unknown = group.get('accounting_unknown_job_ids', [])
        if not isinstance(stage_unknown, list) or not all(isinstance(sid, str) for sid in stage_unknown):
            raise ValueError('Malformed unknown-accounting list')
        unknown.extend(stage_unknown)
        known = group.get('accounting_complete') is True and not stage_unknown and not pending
        if group.get('collection_risks'):
            raise ValueError('Allocation/cap risk is DEFERRED until raw evidence review')
        withheld = group.get('state') == 'NOT_RELEASED_UPSTREAM_FAILURE'
        states[stage] = group.get('state')
        if withheld and (jobs or submitted_ids or pending or not known):
            raise ValueError('Cannot withdraw reservation for a submitted or uncertain stage')
        if pending:
            unknown.append('UNRESOLVED_SUBMISSION_'+stage)
        if not known and not stage_unknown and not pending:
            unknown.append(next(iter(budget_ids), 'UNVERIFIED_STAGE_'+stage))
        if not jobs:
            reserve += 0 if withheld else cap
            continue
        job = jobs[0]
        sid = str(job.get('slurm_job_id', ''))
        if not re.fullmatch(r'[1-9][0-9]*', sid) or sid in ids:
            raise ValueError('Missing or duplicate recovery parent ID')
        ids.add(sid)
        amount = _number(job.get('actual_card_hours'), stage+' parent fee')
        if amount > cap + 1e-6:
            raise ValueError('Stage fee exceeds its registered cap')
        row = job.get('sacct', {})
        zero_pending = (job.get('state') in ('PENDING', 'CONFIGURING', 'CANCELLED') and amount == 0
                        and row.get('elapsed_seconds') == 0 and job.get('allocation_gpu_count') == 0
                        and row.get('alloc_tres') == '')
        if (job.get('hard_gpu_hours') != cap or job.get('gpus') != 1
                or job.get('cpu_cores') != 6 or job.get('memory_GiB') != 90
                or job.get('hard_cap_exceeded')
                or (job.get('allocation_gpu_count', 1) != 1 and not zero_pending)
                or (job.get('unexpected_gpu_allocation') and not zero_pending)):
            raise ValueError('Stage resource/allocation anomaly')
        if known:
            if (str(row.get('job_id', '')) != sid or row.get('state') != job.get('state')
                    or type(row.get('elapsed_seconds')) is not int or row['elapsed_seconds'] < 0):
                raise ValueError('Known parent needs exact saved Slurm accounting binding')
            allocated = job.get('allocation_gpu_count', 1)
            if abs(amount-allocated*row['elapsed_seconds']/3600) > 1e-12:
                raise ValueError('Stored fee disagrees with saved parent accounting; no recalculation repair')
            # A rounded zero-second bill can still have an allocation. Only
            # an explicit unallocated pre-run/cancelled record has no TRES.
            if not zero_pending:
                _verify_allocated_tres(row.get('alloc_tres'))
        actual += amount
        if not known or job.get('state') not in TERMINAL:
            reserve += max(0, cap-amount)
        if not known:
            unknown.append(sid)
        new_rows.append(copy.deepcopy(job))
    for index, stage in enumerate(STAGES):
        if states[stage] == 'NOT_RELEASED_UPSTREAM_FAILURE':
            upstream_failed = any(
                groups[prior].get('accounting_complete') is True
                and not groups[prior].get('accounting_unknown_job_ids')
                and not groups[prior].get('pending_submission')
                and len(groups[prior]['jobs']) == 1
                and groups[prior]['jobs'][0].get('state') in TERMINAL - {'COMPLETED'}
                for prior in STAGES[:index])
            if not upstream_failed:
                raise ValueError('Withheld reservation needs a verified upstream failed parent')
    package_actual = 660/3600 + actual
    total = baseline['actual_card_hours'] + actual
    if package_actual + reserve > 6.5 + 1e-9 or total + reserve > 10 + 1e-9:
        raise ValueError('Recovery exceeds original additional6.5 or root10 pool')
    unknown = sorted(set(unknown))
    package = dict(schema='review10-v22-budget-v1', authorized_card_hours=6.5,
        original_additional_cap_card_hours=6.5, frozen652_actual_card_hours=507/3600,
        frozen656_actual_card_hours=153/3600, historical_package_actual_card_hours=660/3600,
        remaining_cap_after652_656=6.5-660/3600, actual_card_hours=package_actual,
        actual_plus_remaining_caps=package_actual+reserve, new_actual_card_hours=actual,
        remaining_caps=reserve, new_remaining_caps=reserve,
        package_actual_plus_remaining_caps=package_actual+reserve,
        jobs=new_rows, stage_caps=copy.deepcopy(CAPS), stage_states=states,
        accounting_complete=not unknown, accounting_unknown_job_ids=unknown,
        same_original10_pool=True, updated_at_utc=collect.now())
    root = copy.deepcopy(baseline)
    root.update(jobs=copy.deepcopy(baseline['jobs'])+new_rows, actual_card_hours=total,
        actual_gpu_hours=total, remaining_caps=reserve, actual_plus_remaining_caps=total+reserve,
        unallocated_gpu_hours=10-total-reserve, accounting_complete=not unknown,
        accounting_unknown_job_ids=unknown, review10_package_actual_card_hours=package_actual,
        review10_package_cap_card_hours=6.5, review10_package_actual_plus_remaining_caps=package_actual+reserve,
        review10_reservation_card_hours=reserve, review10_recovery_new_actual_card_hours=actual,
        review10_recovery_stage_caps_card_hours=copy.deepcopy(CAPS),
        review10_authorization='claude10/recovery_v22/AUTHORIZATION.json',
        conditional_GRPO_reservation_card_hours=CAPS['grpo'] if not groups['grpo']['jobs']
            and groups['grpo'].get('state') != 'NOT_RELEASED_UPSTREAM_FAILURE' else 0,
        active_approval_reference='REVIEW_CLAUDE_10_v2_sft.md finalsection and direct user continue approval',
        accounting_updated_at_utc=collect.now(), formal_training_released=False,
        billing='Each verified allocation parent once; AllocTRES GPUs times ElapsedRaw/3600; no cross-pool transfer',
        submission_rule='One approved v22 teacher, verified teacher -> SFT76/80 -> unique GRPO19/20; no automatic retries')
    return root, package


def _stage_budget_headers(stage, group):
    """Derive summaries only after the complete groups pass aggregate_budget."""
    cap = CAPS[stage]
    jobs = group['jobs']
    pending = group.get('pending_submission', False)
    known = (group.get('accounting_complete') is True and not group.get('accounting_unknown_job_ids')
             and not pending and not group.get('collection_risks'))
    amount = jobs[0]['actual_card_hours'] if jobs else 0.
    if jobs:
        parent_state = jobs[0]['state']
        remaining = 0. if known and parent_state in TERMINAL else max(0., cap-amount)
        state = parent_state if known else 'ACCOUNTING_UNKNOWN'
    elif pending:
        remaining, state = cap, 'SUBMISSION_UNCERTAIN'
    else:
        remaining = 0. if group.get('state') == 'NOT_RELEASED_UPSTREAM_FAILURE' else cap
        state = group.get('state') if known else 'ACCOUNTING_UNKNOWN'
    return dict(state=state, remaining_caps=remaining,
                actual_plus_remaining_caps=amount+remaining, accounting_complete=known)


def _manifest_job(stage, ledger, manifest, previous):
    if (not isinstance(manifest, dict) or not isinstance(manifest.get('jobs'), list)
            or len(manifest['jobs']) != 1 or len(ledger['jobs']) != 1):
        raise ValueError('Expected one frozen manifest and submitted parent')
    job, declared = ledger['jobs'][0], manifest['jobs'][0]
    if job.get('id') != declared.get('id') or any(job.get(k) != declared.get(k)
            for k in ('gpus', 'cpu_cores', 'memory_GiB', 'hard_gpu_hours')):
        raise ValueError('Submission resource fields differ from frozen manifest')
    if (job.get('gpus') != 1 or job.get('cpu_cores') != 6 or job.get('memory_GiB') != 90
            or job.get('hard_gpu_hours') != CAPS[stage]):
        raise ValueError('Frozen stage resources differ from approved cap')
    remote = Path(manifest.get('remote_smoke_root', ''))
    sid = str(job.get('slurm_job_id', ''))
    if (not remote.is_absolute() or remote.parts[-4:] != ('training_smoke','claude10','recovery_v22',stage)
            or sid in FROZEN_SECONDS or not re.fullmatch(r'[1-9][0-9]*', sid)
            or job.get('remote_output') != str(remote/'outputs'/f'{stage}_{sid}')
            or job.get('remote_log') != str(remote/f'slurm-{sid}.out')):
        raise ValueError('Only the recorded parent output and Slurm log may be retrieved')
    if (not isinstance(previous, dict) or previous.get('authorized_card_hours') != CAPS[stage]
            or not isinstance(previous.get('jobs'), list) or len(previous['jobs']) != 1
            or str(previous['jobs'][0].get('slurm_job_id', '')) != sid
            or previous['jobs'][0].get('id') != job.get('id')):
        raise ValueError('Submitted parent must already be in the stage budget')
    return job


def recover_stage(stage):
    """One query; failures retain raw evidence and fees, never infer job purge."""
    if stage not in STAGES:
        raise ValueError('Unknown recovery stage')
    home = PACKAGE/stage
    ledger = collect.load(home/'SUBMISSIONS.json')
    if not isinstance(ledger, dict) or not isinstance(ledger.get('jobs'), list):
        raise ValueError('Missing or malformed submission ledger')
    if not ledger['jobs']:
        return  # An unresolved intent stays reserved; never submit or invent its ID.
    manifest, previous = collect.load(home/'JOB_MANIFEST.json'), collect.load(home/'BUDGET.json')
    sid = str(ledger['jobs'][0].get('slurm_job_id', ''))
    try:
        job = _manifest_job(stage, ledger, manifest, previous)
        snapshot = collect.query(home, manifest, 'your-cluster')
        if snapshot['account']['code'] != 0:
            raise RuntimeError('sacct query failed; original observation retained in LATEST.json')
        reconciled = snapshot.get('ledger', {})
        if reconciled.get('ledger_status') != 'RECONCILED':
            raise RuntimeError('Remote submission ledger is unknown; no reservation release')
        records, risks, unknown = collect.parse_accounting(snapshot['account']['stdout'], manifest,
            ledger, previous, snapshot.get('allocation_proofs'))
        for record_id, row in records.items():
            zero_unallocated = (row.get('state') in ('PENDING', 'CONFIGURING', 'CANCELLED')
                                and row.get('elapsed_seconds') == 0
                                and row.get('allocated_gpus') == 0 and row.get('alloc_tres') == '')
            if row.get('verified') and not zero_unallocated:
                try:
                    _verify_allocated_tres(row.get('alloc_tres'))
                except ValueError as error:
                    risks.append('observed_allocation_conflict:'+record_id+':'+str(error))
        if risks:
            raise ValueError('Resource/cap anomaly requires raw evidence review: '+','.join(risks))
        budget = collect.reconcile_budget(home, ledger, records, unknown, str(home/'LATEST.json'))
        # The legacy helper defaults blank TRES to requested GPUs. Retain the
        # current parser's explicit zero allocation without rewriting its raw
        # accounting fields or fee, even for a cancellation before allocation.
        for saved_job in budget['jobs']:
            saved_sid = str(saved_job.get('slurm_job_id', ''))
            observed = records.get(saved_sid, {})
            if (observed.get('verified') is True and saved_sid not in unknown
                    and observed.get('state') in ('PENDING', 'CONFIGURING', 'CANCELLED')
                    and observed.get('elapsed_seconds') == 0
                    and observed.get('allocated_gpus') == 0 and observed.get('alloc_tres') == ''):
                raw = saved_job.get('sacct', {})
                if (saved_job.get('state') != observed['state']
                        or _number(saved_job.get('actual_card_hours'), 'unallocated fee') != 0
                        or str(raw.get('job_id', '')) != saved_sid
                        or raw.get('state') != observed['state'] or raw.get('elapsed_seconds') != 0
                        or raw.get('alloc_tres') != ''):
                    raise ValueError('Reconciled zero allocation differs from verified raw parent evidence')
                saved_job['allocation_gpu_count'] = 0
                saved_job['unexpected_gpu_allocation'] = saved_job.get('gpus') != 0
        budget.pop('collection_error', None)
        budget.pop('collection_risks', None)
        atomic_json(home/'BUDGET.json', budget)
        account = records.get(sid, {})
        terminal, retrieved = account.get('state') in TERMINAL, False
        out, result = home/'outputs'/f'{stage}_{sid}', None
        if terminal:
            out.parent.mkdir(parents=True, exist_ok=True)
            for remote in (job['remote_output'], job['remote_log']):
                subprocess.run(['scp', '-r', 'your-cluster:'+remote,
                    str(out.parent if remote == job['remote_output'] else out)],
                    check=True, capture_output=True, timeout=120)
            result = collect.load(out/'result.json')
            retrieved = out.is_dir() and (out/Path(job['remote_log']).name).is_file()
            atomic_json(out/'RETRIEVED_TERMINAL.json', dict(at_utc=collect.now(), state=account['state'], retrieved=retrieved))
        atomic_json(home/'ANALYSIS.json', dict(checked_at_utc=collect.now(),
            collection_status='DEFERRED' if unknown else 'COLLECTED', accounting_unknown_job_ids=unknown,
            risks=[], jobs=[dict(job, state=account.get('state', 'ACCOUNTING_UNKNOWN'), accounting=account,
                terminal=terminal, terminal_outputs_retrieved=retrieved, result=collect.json_evidence(result),
                independently_verified=False)], all_terminal_recovered=terminal and retrieved))
    except Exception as error:
        budget = collect.load(home/'BUDGET.json', previous)
        if isinstance(budget, dict):
            budget.update(accounting_complete=False, accounting_unknown_job_ids=[sid],
                          collection_error=type(error).__name__+': '+str(error))
            if 'risks' in locals() and risks:
                budget['collection_risks'] = risks
            atomic_json(home/'BUDGET.json', budget)
        failure = dict(collection_status='DEFERRED', checked_at_utc=collect.now(),
            error=type(error).__name__+': '+str(error), raw_evidence='LATEST.json',
            accounting_unknown_job_ids=[sid], risks=locals().get('risks', []))
        atomic_json(home/'COLLECTION_FAILURE.json', failure)
        atomic_json(home/'ANALYSIS.json', failure)


def read_groups():
    result = {}
    for stage in STAGES:
        group, ledger = collect.load(PACKAGE/stage/'BUDGET.json'), collect.load(PACKAGE/stage/'SUBMISSIONS.json')
        if (not isinstance(group, dict) or not isinstance(ledger, dict)
                or not isinstance(ledger.get('jobs'), list) or len(ledger['jobs']) > 1):
            raise ValueError('Malformed stage budget/submission ledger')
        if not all(isinstance(job, dict) for job in ledger['jobs']):
            raise ValueError('Malformed stage submission parent')
        group['submitted_job_ids'] = [str(j.get('slurm_job_id', '')) for j in ledger['jobs']]
        if ledger['jobs'] and group.get('jobs') and ledger['jobs'][0].get('id') != group['jobs'][0].get('id'):
            raise ValueError('Stage ledger/budget logical job IDs differ')
        marker = ledger.get('pending_submission', False)
        if marker is not None and not isinstance(marker, bool) and not (isinstance(marker, dict) and marker):
            raise ValueError('Malformed stage submission intent marker')
        group['pending_submission'] = bool(marker)
        result[stage] = group
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--collect', action='store_true', help='One read-only query per submitted stage')
    action.add_argument('--publish', action='store_true', help='Publish saved local fees; no remote query')
    parser.add_argument('--stage', choices=STAGES)
    args = parser.parse_args(argv)
    if args.publish and args.stage:
        parser.error('--stage applies only to --collect')
    authorization = collect.load(PACKAGE/'AUTHORIZATION.json')
    validate_authorization(authorization)
    validate_frozen_history()
    baseline = collect.load(PACKAGE/'BASELINE_BUDGET.json')
    # Validate all local bindings before any query or publication.
    before_query = read_groups()
    # A caller may explicitly request one new read-only observation after
    # reviewing DEFERRED evidence. Keep the persisted risk until a valid new
    # observation clears it; prevalidate all other local bindings/caps first.
    for group in before_query.values():
        group.pop('collection_risks', None)
    aggregate_budget(baseline, authorization, before_query)
    if args.collect:
        for stage in ((args.stage,) if args.stage else STAGES):
            recover_stage(stage)
    latest_groups = read_groups()
    try:
        root, package = aggregate_budget(baseline, authorization, latest_groups)
    except ValueError as error:
        failure = dict(collection_status='DEFERRED', error=str(error), checked_at_utc=collect.now())
        atomic_json(PACKAGE/'COLLECTION_FAILURE.json', failure)
        # A fresh risk revokes terminal-fee certainty, including any previously
        # released unused cap. Keep stored parent fees; never repair them from
        # conflicting raw observations. Remove risks only in this arithmetic
        # copy, leaving all persisted raw evidence and risk markers intact.
        affected = set()
        conservative_groups = copy.deepcopy(latest_groups)
        for group in conservative_groups.values():
            affected.update(group.get('accounting_unknown_job_ids', []))
            if group.pop('collection_risks', None):
                parent_ids = [str(j['slurm_job_id']) for j in group['jobs']]
                affected.update(parent_ids)
                group.update(accounting_complete=False, accounting_unknown_job_ids=sorted(
                    set(group.get('accounting_unknown_job_ids', [])) | set(parent_ids)))
        try:
            deferred_root, deferred_package = aggregate_budget(baseline, authorization, conservative_groups)
        except (ValueError, KeyError, TypeError):
            # Unsafe bindings/resources cannot support stage arithmetic. Freeze
            # each ledger's entire remaining envelope around its last saved fee.
            deferred_root = deferred_package = None
        for path, fresh, cap in ((PACKAGE/'BUDGET.json', deferred_package, 6.5),
                                 (HOME/'BUDGET.json', deferred_root, 10)):
            previous = collect.load(path)
            if isinstance(previous, dict):
                if fresh is not None:
                    previous.update(fresh)
                else:
                    amount = _number(previous.get('actual_card_hours'), 'last published fee')
                    reserve = max(0., cap-amount)
                    previous.update(remaining_caps=reserve, actual_plus_remaining_caps=amount+reserve,
                                    unallocated_gpu_hours=0, spending_frozen=True)
                    if cap == 6.5:
                        previous.update(new_remaining_caps=reserve,
                                        package_actual_plus_remaining_caps=amount+reserve)
                    else:
                        package_fee = _number(previous.get('review10_package_actual_card_hours'),
                                              'last published package fee')
                        previous.update(review10_reservation_card_hours=max(0., 6.5-package_fee),
                                        review10_package_actual_plus_remaining_caps=6.5)
                unknown = sorted(set(previous.get('accounting_unknown_job_ids', [])) | affected)
                previous.update(accounting_complete=False, accounting_unknown_job_ids=unknown,
                    collection_status='DEFERRED', collection_error=str(error))
                atomic_json(path, previous)
        raise
    analysis = collect.load(HOME/'ANALYSIS.json', {})
    if not isinstance(analysis, dict):
        raise ValueError('Malformed root analysis; no metadata published')
    # The generic legacy helper retains stale stage headers. Publish only
    # summaries derived from the fully validated saved parent records. Keep
    # fees, raw sacct fields, parent identities and submission ledgers intact.
    stage_budgets = {}
    for stage, group in latest_groups.items():
        original = collect.load(PACKAGE/stage/'BUDGET.json')
        bound = copy.deepcopy(original)
        bound.update(submitted_job_ids=group['submitted_job_ids'],
                     pending_submission=group['pending_submission'])
        if bound != group:
            raise ValueError('Stage budget changed during validated publication: '+stage)
        stage_budgets[stage] = dict(original, **_stage_budget_headers(stage, group))
    package['stage_states'] = {stage:budget['state'] for stage,budget in stage_budgets.items()}
    analysis.update(review10_v22_recovery={s:collect.load(PACKAGE/s/'ANALYSIS.json', {}) for s in STAGES},
                    review10_v22_updated_at_utc=collect.now())
    for stage, budget in stage_budgets.items():
        atomic_json(PACKAGE/stage/'BUDGET.json', budget)
    atomic_json(PACKAGE/'BUDGET.json', package)
    atomic_json(HOME/'BUDGET.json', root)
    atomic_json(HOME/'ANALYSIS.json', analysis)
    print(json.dumps({k:root[k] for k in ('actual_card_hours', 'remaining_caps',
        'actual_plus_remaining_caps', 'unallocated_gpu_hours', 'accounting_complete')}))


if __name__ == '__main__':
    main()
