"""Refresh local Review07 control status from collected evidence, without SSH."""
import datetime
import json
from pathlib import Path

from claude07_safety import TERMINAL, atomic_json

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def update(root=ROOT):
    status_path = root/'calibration/REVIEW_STATUS.json'
    status = load(status_path)
    gpu_limit = status.get('max_concurrent_gpu', 2)
    if type(gpu_limit) is not int or gpu_limit not in (1, 2):
        raise ValueError('Review07 GPU concurrency must be 1 or 2')
    pools = {}
    all_rows = []
    for name, relative, expected in [('CPU', 'calibration/claude07', 1),
                                     ('DFT_GPU', 'calibration_gpu/claude07', 8),
                                     ('TRAINING_SMOKE', 'training_smoke', 2)]:
        home = root/relative
        ledger = load(home/'SUBMISSIONS.json')
        analysis = load(home/'ANALYSIS.json')
        rows = analysis.get('jobs', [])
        ids = [str(j['slurm_job_id']) for j in ledger.get('jobs', [])]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate submitted IDs: '+name)
        complete_list = (len(ids) == expected and len(rows) == expected and
            {str(j['slurm_job_id']) for j in rows} == set(ids) and 'pending_submission' not in ledger)
        terminal = complete_list and all(j.get('state') in TERMINAL for j in rows)
        recovered = terminal and all(j.get('terminal_outputs_retrieved', False) for j in rows)
        pools[name] = dict(job_ids=ids, expected_jobs=expected,
            all_terminal=terminal, all_terminal_outputs_retrieved=recovered,
            budget=load(home/'BUDGET.json'),
            jobs=[{k:j.get(k) for k in ['id', 'name', 'slurm_job_id', 'state', 'usable']} for j in rows])
        all_rows.extend(rows)
    all_terminal = all(p['all_terminal'] for p in pools.values())
    recovered = all(p['all_terminal_outputs_retrieved'] for p in pools.values())
    budgets_verified = all(pools[name]['budget'].get('safe') is True for name in ['CPU', 'DFT_GPU'])
    smoke_budget = pools['TRAINING_SMOKE']['budget']
    budgets_verified = bool(budgets_verified and smoke_budget.get('accounting_complete') is True
                            and not smoke_budget.get('pool_cap_exceeded', True))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    review08 = load(root/'training_smoke/REVIEW08_AUTHORIZATION.json')
    grpo = review08.get('conditional_grpo', {})
    submission_scope = 'NONE; all 9 DFT and 2 approved smoke jobs submitted once'
    if grpo.get('submissions_allowed') == 1 and grpo.get('cap_card_hours') == 2:
        submission_scope = ('Original 11 jobs locked; only one Review08 GRPO smoke <=2 card-hours '
                            'may be prepared/submitted after 633/635 evidence gates, per REVIEW08_AUTHORIZATION.json')
    current = dict(checked_at_utc=now, approval_reference='REVIEW_CLAUDE_07_gates.md',
        pools=pools, all_11_jobs_terminal=all_terminal, all_11_outputs_retrieved=recovered,
        budgets_verified=budgets_verified, ready_for_final_audit=recovered and budgets_verified,
        formal_training_allowed=False, further_submissions_allowed=False,
        no_automatic_retries=True, max_concurrent_gpu=gpu_limit)
    atomic_json(root/'calibration/claude07/STATUS.json', current)
    status.update(approval_reference=current['approval_reference'],
        p0_state='CLAUDE07_RECOVERED_AWAITING_MANUAL_REVIEW' if recovered and budgets_verified else 'CLAUDE07_RUNNING_OR_RECOVERING',
        claude07_status='calibration/claude07/STATUS.json',
        claude07_cpu_job_ids=pools['CPU']['job_ids'],
        claude07_gpu_job_ids=pools['DFT_GPU']['job_ids'],
        training_smoke_job_ids=pools['TRAINING_SMOKE']['job_ids'],
        claude07_all_terminal=all_terminal, claude07_all_outputs_retrieved=recovered,
        claude07_ready_for_final_audit=recovered and budgets_verified,
        active_current_job_ids=sorted({sid for p in pools.values() for sid in p['job_ids']} -
            {str(j['slurm_job_id']) for j in all_rows if j.get('state') in TERMINAL}, key=int),
        further_submissions_allowed=False, gpu_further_submissions_allowed=False,
        formal_training_allowed=False, last_claude07_collection_at_utc=now,
        further_submissions_scope=submission_scope,
        awaiting='Final evidence audit, measured training estimates and manual review' if recovered else
                 'Collect existing 9 DFT + 2 smoke jobs; no new computation or retries',
        cumulative_actual_core_hours=pools['CPU']['budget'].get('actual'),
        actual_gpu_card_hours=pools['DFT_GPU']['budget'].get('actual'),
        actual_training_smoke_card_hours=pools['TRAINING_SMOKE']['budget'].get('actual_card_hours'))
    atomic_json(status_path, status)
    return dict(all_11_jobs_terminal=all_terminal, all_11_outputs_retrieved=recovered,
                job_ids={name:pool['job_ids'] for name,pool in pools.items()})


if __name__ == '__main__':
    print(json.dumps(update(), indent=2))
