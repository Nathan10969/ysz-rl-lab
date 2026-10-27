"""Fixed 16×8 Review10 teacher; real physics loads only inside an authorized job."""
from __future__ import annotations
import time
PROCESS_STARTED_AT = time.monotonic()
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import numpy as np
import claude08_env as env
from claude08_data import PhysicalReceipt, seed_streams
from calibration import make_state
from claude10_data import (SEED, PURPOSE, MIX, PHYSICS_SOURCE_SHA, serialize_result,
                            serialize_receipt, source_hashes, write_json)
from claude10_data import persist_result

LIMITS = env.BudgetLimits(max_relax_actual=240, max_raw_ef=72240, wall_seconds=6600)
TEMPERATURES = tuple(map(float, np.linspace(.20, .02, 8)))


def prepare_output(output):
    """The backend may have created its empty log directory, but no run may be reused."""
    output = Path(output)
    if output.exists():
        if (not output.is_dir() or {p.name for p in output.iterdir()} != {'physics'}
                or not (output/'physics').is_dir() or any((output/'physics').iterdir())):
            raise FileExistsError(output)
    else: output.mkdir(parents=True, exist_ok=False)


def evaluate_batch(physics, requests, counter, records, *, metadata=None, journal=None):
    """Charge every attempted lane before physics; persist failed/unconverged outputs."""
    requests = list(requests)
    if not 1 <= len(requests) <= 4: raise ValueError('Batch size must be 1..4')
    counter.check_time()
    if counter.counters.relax_actual + len(requests) > 240:
        raise env.BudgetExceeded('Teacher actual relaxation cap')
    before = counter.counters.raw_ef
    counter.counters.relax_actual += len(requests)
    begin = counter.clock()
    entries = []
    for lane, request in enumerate(requests):
        entry = {'id':len(records), 'status':'pending', 'request_key':request.key,
                 'request_occupancy':list(request.occupancy), 'accepted':False,
                 **({} if metadata is None else metadata[lane])}
        records.append(entry)
        entries.append(entry)
    results = []
    try:
        results = list(physics.relax_many(requests, counter, max_steps=300))
        if len(results) != len(requests): raise ValueError('Physics batch returned wrong lane count')
        raw = [getattr(result, 'evidence', {}).get('raw_ef_attempts') for result in results]
        if (any(type(n) is not int or not 1 <= n <= 301 for n in raw)
                or sum(raw) != counter.counters.raw_ef-before):
            raise ValueError('Per-lane raw E/F evidence does not reconcile with charged counter')
        for entry, request, result in zip(entries, requests, results):
            entry['result'] = serialize_result(result)
            entry['physics_evidence'] = dict(result.evidence)
        for entry, request, result in zip(entries, requests, results):
            evidence = entry['physics_evidence']
            if (evidence.get('formal_receipt') is not True or evidence.get('status') != 'converged'
                    or type(evidence.get('nsteps')) is not int or not 0 <= evidence['nsteps'] <= 300):
                raise ValueError('Only converged full BFGS receipts may enter teacher state')
            env.verify_result(result, request.spec)
            receipt = PhysicalReceipt.verify(request, result, relax_actual=1, raw_ef=evidence['raw_ef_attempts'])
            entry['receipt'] = serialize_receipt(receipt)
            entry['status'] = 'completed'
        counter.check_time()
        return entries
    except Exception as exc:
        evidence = getattr(physics, 'last_evidence', [])
        for lane, entry in enumerate(entries):
            entry.update(status='failed', error=type(exc).__name__+': '+str(exc))
            if lane < len(evidence): entry.setdefault('physics_evidence', dict(evidence[lane]))
            if lane < len(results): entry.setdefault('result', serialize_result(results[lane]))
        raise
    finally:
        for entry in entries:
            entry.update(operation_seconds=counter.clock()-begin, counters_after=asdict(counter.counters))
            if journal is not None:
                with Path(journal).open('a') as handle: handle.write(json.dumps(entry, allow_nan=False)+'\n')


def _uniform(occupancy, rng):
    legal = env.enumerate_legal_actions(occupancy)
    return legal[int(rng.integers(len(legal)))]


def breed_population(occupancy, parents, rng):
    """Endpoint crossover, one-endpoint mutation, then nearest legal genome repair."""
    legal = env.enumerate_legal_actions(occupancy)
    children = []
    for index in range(4):
        left, right = parents[index % 2], parents[(index+1) % 2]
        a = env.parse_action(json.dumps(left['action']))
        b = env.parse_action(json.dumps(right['action']))
        op = a.op if index % 2 == 0 else b.op
        crossed = [a.sites[int(rng.integers(2))], b.sites[int(rng.integers(2))]]
        mutant = list(crossed)
        sublattice = np.flatnonzero(env._parent().numbers == (40 if op=='swap_cations' else 8))
        endpoint = int(rng.integers(2))
        mutant[endpoint] = int(sublattice[int(rng.integers(len(sublattice)))])
        candidates = [action for action in legal if action.op == op]
        score = [len(set(mutant) & set(action.sites)) for action in candidates]
        best = max(score)
        repair = [action for action,s in zip(candidates,score) if s == best]
        action = repair[int(rng.integers(len(repair)))]
        children.append({'action':json.loads(action.to_json()), 'parent_ids':[left['id'],right['id']],
            'crossover_sites':crossed, 'mutated':True, 'mutation_endpoint':endpoint,
            'mutated_sites':mutant, 'repaired':sorted(mutant) != list(action.sites)})
    return children


def run_trajectory(mode, initial, physics, counter, records, *, rng, trajectory_id, journal=None):
    if mode not in MIX: raise ValueError('Unknown fixed teacher mode')
    current = initial
    trajectory = {'id':trajectory_id, 'mode':mode, 'initial_id':initial['id'], 'steps':[],
                  'seed':int(rng.integers(0, 2**63-1)), 'complete':False}
    population = None
    for step in range(8):
        counter.check_time()
        occupancy = current['receipt']['terminal_occupancy']
        if mode == 'GA':
            if population is None:
                legal = env.enumerate_legal_actions(occupancy)
                selected = rng.choice(len(legal), 4, replace=False)
                population = [{'action':json.loads(legal[int(i)].to_json())} for i in selected]
            actions = [env.parse_action(json.dumps(child['action'])) for child in population]
        else: actions = [_uniform(occupancy, rng)]
        requests = [env.build_request(env.apply_action(occupancy, action)) for action in actions]
        metadata = [{'trajectory_id':trajectory_id, 'step':step, 'before_id':current['id'],
                     'action':json.loads(action.to_json())} for action in actions]
        candidates = evaluate_batch(physics, requests, counter, records, metadata=metadata, journal=journal)
        counter.counters.attempts += 1
        chosen = min(candidates, key=lambda record:(record['receipt']['summary']['energy_eV'],record['id']))
        delta = chosen['receipt']['summary']['energy_eV'] - current['receipt']['summary']['energy_eV']
        uniform = float(rng.random()) if mode == 'SA' else None
        accepted = mode != 'SA' or delta <= 0 or uniform < np.exp(-delta/TEMPERATURES[step])
        row = {'step':step, 'before_id':current['id'], 'candidate_ids':[c['id'] for c in candidates],
               'accepted_id':chosen['id'] if accepted else None, 'energy_delta_eV':delta}
        if mode == 'SA': row.update(temperature_eV=TEMPERATURES[step], acceptance_uniform=uniform)
        if accepted:
            chosen['accepted'] = True
            current = chosen
        if mode == 'GA':
            parents = sorted(candidates, key=lambda c:(c['receipt']['summary']['energy_eV'],c['id']))[:2]
            population = breed_population(current['receipt']['terminal_occupancy'], parents, rng)
            row.update(selected_parent_ids=[p['id'] for p in parents], breeding=population)
        trajectory['steps'].append(row)
    trajectory.update(complete=True, terminal_id=current['id'])
    return trajectory


def collect_teacher(physics, *, output=None, started_at=None, clock=time.monotonic, initial_occupancies=None):
    """Fixed collector owning backend cleanup. Nonproduction backends are diagnostic."""
    from claude10_physics import BatchMACEPhysics
    real = type(physics) is BatchMACEPhysics and physics.device == 'cuda' and initial_occupancies is None
    started_at = clock() if started_at is None else started_at
    output = None if output is None else Path(output)
    if output is not None: prepare_output(output)
    counter = env.RawEFCounter(env.Counters(), LIMITS, started_at=started_at, clock=clock)
    proof = {'schema':'review10-teacher-v1', 'status':'running', 'seed':SEED, 'purpose':PURPOSE,
             'diagnostic':not real, 'runtime':{'GPU_work_performed':real}, 'source_hashes':source_hashes(),
             'limits':asdict(LIMITS), 'trajectories':[], 'records':[], 'temperature_eV':TEMPERATURES,
             'physical_protocol':'BFGS/fmax0.03/steps300/maxstep0.1/float64'}
    journal = None if output is None else output/'raw_candidates.jsonl'
    rng = np.random.default_rng(seed_streams(SEED)[0])
    try:
        if real:
            from smoke_grpo08 import require_single_gpu_job
            proof['runtime'].update(require_single_gpu_job())
        if real and proof['source_hashes']['claude10_physics.py'] != PHYSICS_SOURCE_SHA:
            raise ValueError('Reviewed physics source SHA changed')
        occupancies, seeds, keys = [], [], set()
        for _ in range(1000):
            seed = int(rng.integers(0, 2**63-1))
            occupancy = env.validate_occupancy(make_state(seed,3,16).numbers)
            key = env.occupancy_key(occupancy)
            if key in keys: continue
            keys.add(key); occupancies.append(occupancy); seeds.append(seed)
            if len(occupancies) == 16: break
        if initial_occupancies is not None:
            occupancies = [env.validate_occupancy(o) for o in initial_occupancies]
            if len(occupancies) != 16 or len({env.occupancy_key(o) for o in occupancies}) != 16:
                raise ValueError('Sixteen distinct initial canonical states required')
        proof['initial_seeds'] = seeds
        initials = []
        for start in range(0,16,4):
            batch = evaluate_batch(physics, [env.build_request(o) for o in occupancies[start:start+4]],
                                   counter, proof['records'], journal=journal)
            for entry in batch: entry['accepted'] = True
            initials.extend(batch)
        for index, mode in enumerate(MIX):
            trajectory = run_trajectory(mode, initials[index], physics, counter, proof['records'],
                                       rng=rng, trajectory_id=index, journal=journal)
            proof['trajectories'].append(trajectory)
            if output is not None: write_json(output/'progress.json', {**proof, 'counters':asdict(counter.counters)})
        counter.check_time()
        proof.update(status='completed', counters=asdict(counter.counters))
        from claude10_data import verify_teacher_result
        verified = verify_teacher_result(proof)
        counter.check_time()
        families = [r['receipt']['summary']['terminal_key'] for r in proof['records']]
        proof.update(accepted_sft_rows=sum(len(t['rows']) for t in verified),
                     unique_terminal_cells=len(set(families)), repeated_terminal_cells=len(families)-len(set(families)))
    except Exception as exc:
        proof.update(status='failed', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        cleanup_begin = clock()
        cleanup_error = None
        try:
            if callable(getattr(physics,'close',None)): physics.close()
            proof['cleanup'] = {'status':'completed','seconds':clock()-cleanup_begin}
        except Exception as exc:
            cleanup_error = exc
            proof.update(status='failed',error='Backend cleanup failed: '+str(exc),
                         cleanup={'status':'failed','seconds':clock()-cleanup_begin})
        proof.update(counters=asdict(counter.counters), wall_seconds=clock()-started_at)
        overrun = proof['wall_seconds'] >= 6600
        if output is not None:
            overrun = persist_result(output/'result.json',proof,started_at=started_at,clock=clock)
        elif overrun: proof.update(status='failed',error='Entire-stage runtime cap reached')
        if cleanup_error is not None: raise RuntimeError('Teacher backend cleanup failed') from cleanup_error
        if overrun: raise env.BudgetExceeded('Teacher entire-stage runtime cap including cleanup/persist')
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    from smoke_grpo08 import require_single_gpu_job
    runtime = require_single_gpu_job()
    from claude10_physics import BatchMACEPhysics
    engine = BatchMACEPhysics(args.checkpoint, args.output/'physics', device='cuda')
    proof = collect_teacher(engine, output=args.output, started_at=PROCESS_STARTED_AT)
    print(json.dumps({'status':proof['status'], 'counters':proof['counters'], 'runtime':runtime}))


if __name__ == '__main__': main()
