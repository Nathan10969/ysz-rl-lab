"""Recover the section8 supplement; never submit, prepare, retry or cancel jobs.

Original633/635 records remain frozen. New parents have separate one-shot
ledgers, while this aggregates every parent once into the original10h pool.
"""
import argparse
import copy
import json
import math
import hashlib
from pathlib import Path
import subprocess

from claude07_safety import TERMINAL, atomic_json
import collect_claude07_smoke as collect

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT/'training_smoke'
ORIGINAL = HOME/'evidence/review08_section8_before_fix/training_smoke/BUDGET.json'


def reusable_outputs(home, sid):
    proof=collect.load(home/'INDEPENDENT_VERIFICATION.json',{})
    if proof.get('independently_verified') is not True or str(proof.get('job_id'))!=sid:
        return False
    hashes=proof.get('raw_sha256',{})
    if not hashes: return False
    for name,digest in hashes.items():
        if Path(name).name=='RETRIEVED_TERMINAL.json': continue
        path=home/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            return False
    return True


def combined_budget(original, groups, conditional=2):
    if conditional not in (0, 2):
        raise ValueError('Only the original conditional2h GRPO reservation is authorized')
    rows = copy.deepcopy(original['jobs'])
    ids = {str(j['slurm_job_id']) for j in rows}
    if len(ids) != len(rows): raise ValueError('Duplicate original parent ID')
    actual = sum(j['actual_card_hours'] for j in rows)
    reserve = conditional
    complete = original.get('accounting_complete') is True
    unknown=list(original.get('accounting_unknown_job_ids',[]))
    for group in groups:
        complete = complete and group.get('accounting_complete') is True
        for j in group['jobs']:
            sid = j.get('slurm_job_id')
            if sid is not None:
                if str(sid) in ids: raise ValueError('Parent would be billed twice')
                ids.add(str(sid))
                if group.get('accounting_complete') is not True:
                    unknown.append(str(sid))
            amount = j.get('actual_card_hours', 0)
            cap = j['hard_gpu_hours']
            if not math.isfinite(amount) or amount < 0 or cap not in (.75, 2) or amount > cap+1e-6:
                raise ValueError('Supplemental parent budget/allocation anomaly')
            actual += amount
            if j.get('state') not in TERMINAL or group.get('accounting_complete') is not True:
                reserve += max(0, cap-amount)
            rows.append(copy.deepcopy(j))
    if actual+reserve > 10+1e-9: raise ValueError('Original10 card-hour pool would be exceeded')
    result = copy.deepcopy(original)
    result.update(jobs=rows, actual_card_hours=actual, actual_gpu_hours=actual,
                  actual_plus_remaining_caps=actual+reserve, remaining_caps=reserve,
                  conditional_GRPO_reservation_card_hours=conditional,
                  accounting_complete=complete, unallocated_gpu_hours=10-actual-reserve,
                  accounting_unknown_job_ids=sorted(set(unknown)),
                  original633_635_hard_caps_card_hours=1.25,
                  planned_hard_gpu_hours=sum(j.get('hard_gpu_hours',0) for j in rows),
                  accounting_updated_at_utc=collect.now(),
                  active_approval_reference='REVIEW_CLAUDE_08_RL_arm.md section8 plus direct user authorization',
                  original633_635_actual_card_hours=sum(j['actual_card_hours'] for j in original['jobs']),
                  supplemental_parents_separately_ledgered=True, no_third_MACE_attempt=True,
                  billing='Verified Slurm allocation parents only; original635 startup failure remains charged',
                  formal_training_released=False)
    return result


def recover_group(home):
    manifest = collect.load(home/'JOB_MANIFEST.json')
    ledger = collect.load(home/'SUBMISSIONS.json')
    if not ledger or not ledger.get('jobs'): return collect.load(home/'BUDGET.json')
    snapshot = collect.query(home, manifest, 'your-cluster')
    if snapshot['account']['code'] != 0:
        atomic_json(home/'ANALYSIS.json', {'collection_status':'DEFERRED','evidence':'LATEST.json'})
        return collect.load(home/'BUDGET.json')
    previous = collect.load(home/'BUDGET.json')
    records, risks, unknown = collect.parse_accounting(
        snapshot['account']['stdout'], manifest, ledger, previous, snapshot.get('allocation_proofs'))
    budget = collect.reconcile_budget(home, ledger, records, unknown, str(home/'LATEST.json'))
    jobs = []
    for j in ledger['jobs']:
        sid = str(j['slurm_job_id']); account = records.get(sid, {})
        terminal = account.get('state') in TERMINAL
        out = home/'outputs'/('mace_'+sid if home.name=='mace_retry08' else 'grpo_'+sid)
        recovered = False
        result = None
        check = {'valid':False,'errors':['not recovered']}
        if terminal:
            out.parent.mkdir(parents=True, exist_ok=True)
            if not reusable_outputs(home,sid):
                for remote in [j['remote_output'], j['remote_log']]:
                    subprocess.run(['scp','-r','your-cluster:'+remote,str(out.parent if remote==j['remote_output'] else out)],
                                   check=True,capture_output=True,timeout=120)
            result = collect.load(out/'result.json')
            recovered = out.is_dir() and (out/Path(j['remote_log']).name).is_file()
            if result and home.name=='mace_retry08':
                meta = dict(manifest['jobs'][0], id='c07_smoke_mace')
                check = collect.validate_result(meta, result, sid)
            elif result:
                ok = result.get('status') in ('COMPLETED_ONE_UPDATE','NO_UPDATE_ZERO_GROUP_VARIANCE','NOT_RELEASED_LEGALITY_GATE')
                check = {'valid':ok,'errors':[] if ok else ['GRPO did not complete a permitted outcome']}
            atomic_json(out/'RETRIEVED_TERMINAL.json', {'at_utc':collect.now(),'state':account.get('state'),'retrieved':recovered})
        jobs.append(dict(j, state=account.get('state','ACCOUNTING_UNKNOWN'), accounting=account,
                         terminal=terminal, terminal_outputs_retrieved=recovered,
                         usable=bool(account.get('verified') and account.get('state')=='COMPLETED'
                                     and account.get('exit_code')=='0:0' and check['valid']),
                         result_check=check, result=result))
    atomic_json(home/'ANALYSIS.json', {'checked_at_utc':collect.now(),'collection_status':'COLLECTED' if not unknown else 'DEFERRED',
                'accounting_unknown_job_ids':unknown,'risks':risks,'jobs':jobs,
                'all_terminal_recovered':all(j['terminal'] and j['terminal_outputs_retrieved'] for j in jobs)})
    if risks: raise RuntimeError('Supplemental accounting risks: '+repr(risks))
    return budget


def main():
    if (HOME/'claude10/AUTHORIZATION.json').exists():
        raise RuntimeError('Review10 is registered: use review10_control.py; do not overwrite the expanded pool with Review08 aggregation')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true', required=True)
    parser.parse_args()
    auth = collect.load(HOME/'REVIEW08_SUPPLEMENT.json')
    groups = []
    for name in ('mace_retry08','claude08_grpo'):
        if (HOME/name/'BUDGET.json').exists(): groups.append(recover_group(HOME/name))
    budget = combined_budget(collect.load(ORIGINAL), groups, auth['conditional_GRPO_reservation_card_hours'])
    atomic_json(HOME/'BUDGET.json', budget)
    original_analysis=collect.load(HOME/'evidence/review08_section8_before_fix/training_smoke/ANALYSIS.json')
    supplements=[collect.load(HOME/name/'ANALYSIS.json',{}) for name in ('mace_retry08','claude08_grpo')]
    original_analysis.update(review09_supplemental_jobs=[j for a in supplements for j in a.get('jobs',[])],
                             review09_updated_at_utc=collect.now(),review09_pool_budget='BUDGET.json')
    atomic_json(HOME/'ANALYSIS.json',original_analysis)
    print(json.dumps({k:budget[k] for k in ('actual_card_hours','actual_plus_remaining_caps','unallocated_gpu_hours','accounting_complete')}))


if __name__ == '__main__': main()
