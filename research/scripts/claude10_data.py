"""Review10 CPU data contracts. Importing creates no files and loads no model."""
from __future__ import annotations

from dataclasses import asdict
import argparse
import json
import time
from pathlib import Path
import numpy as np
from ase import Atoms

import claude08_env as env
from claude08_data import (PhysicalReceipt, action_sft_row, sha256, unscored_state,
                           split_terminal_families, seed_streams)
from calibration import make_state

SEED = 2026102301
PURPOSE = 'smoke_and_calibration_only_not_formal_D0'
MIX = ['SA'] * 6 + ['GA'] * 4 + ['random'] * 6
PHYSICS_SOURCE_SHA = '6186f0cb1a98909a4be5768bfbc288b494b8ba88796aae2157bfafa1d908ee20'


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name:sha256(root / name) for name in
            ['claude08_env.py', 'claude08_data.py', 'claude10_physics.py',
             'claude10_data.py', 'smoke_teacher10.py', 'smoke_sft10.py']}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def persist_result(path, result, *, started_at, clock=time.monotonic):
    """Check after both result writes so a late write cannot leave a released status."""
    before = clock()
    result['wall_seconds'] = before-started_at
    write_json(path,result)
    after = clock()
    result.update(wall_seconds=after-started_at, result_persist_seconds=after-before)
    overrun = after-started_at >= 6600
    if overrun:
        result.update(status='failed',error='Entire-stage runtime cap reached during cleanup/persist',
                      adapter_release_allowed=False)
    write_json(path,result)
    final = clock()
    if final-started_at >= 6600:
        result.update(status='failed',error='Entire-stage runtime cap reached during cleanup/persist',
                      adapter_release_allowed=False,wall_seconds=final-started_at,
                      result_persist_seconds=final-before)
        write_json(path,result)
        overrun = True
    return overrun


def serialize_result(result):
    atoms = result.atoms
    value = {'numbers':atoms.numbers.tolist(), 'positions':atoms.positions.tolist(),
            'cell':atoms.cell.array.tolist(), 'pbc':atoms.pbc.tolist(),
            'energy_eV':float(result.energy), 'forces':np.asarray(result.forces).tolist(),
            'converged':result.converged}
    def finite_json(item):
        if isinstance(item, dict): return {k:finite_json(v) for k,v in item.items()}
        if isinstance(item, list): return [finite_json(v) for v in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return 'NaN' if np.isnan(item) else ('Infinity' if item > 0 else '-Infinity')
        return item
    return finite_json(value)


def serialize_receipt(receipt):
    receipt.reverify()
    return {'request_occupancy':list(receipt._request.occupancy),
            'spec':asdict(receipt._request.spec), 'summary':receipt.summary(),
            'terminal_occupancy':list(env.map_terminal(receipt._result.atoms).occupancy),
            'result':serialize_result(receipt._result)}


def deserialize_receipt(payload):
    """Reconstruct atoms, rerun mapping/convergence, and compare every claimed field."""
    try:
        result = payload['result']
        atoms = Atoms(numbers=result['numbers'], positions=result['positions'],
                      cell=result['cell'], pbc=result['pbc'])
        request = env.build_request(payload['request_occupancy'], env.RelaxationSpec(**payload['spec']))
        receipt = PhysicalReceipt.verify(request, env.PhysicsResult(atoms, result['energy_eV'],
                    np.asarray(result['forces']), result['converged']), **payload['summary']['costs'])
        if receipt.summary() != payload['summary']:
            raise ValueError('Serialized physical receipt metadata changed')
        if list(env.map_terminal(atoms).occupancy) != payload['terminal_occupancy']:
            raise ValueError('Serialized terminal occupation changed')
        return receipt
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Complete actual physical result and receipt required') from exc


def _state(receipt, counters, history, step, families, seed, best_energy):
    mapping = env.map_terminal(receipt._result.atoms)
    entries = [env.HistoryEntry(h['action'], h['status'], h['delta'], None, None, 0.) for h in history]
    return env.EpisodeState(mapping.occupancy, receipt._result.atoms.copy(), receipt.energy,
            best_energy, mapping.canonical_key, mapping.geometry_hash, frozenset(), set(families),
            env.BudgetLimits(max_relax_actual=240, max_raw_ef=72240), counters,
            ('teacher10',seed), 0., lambda:0., steps_taken=step, history=entries)


def verify_teacher_result(payload):
    """Recover every raw receipt and independently rebuild all accepted connections."""
    if (payload.get('schema') != 'review10-teacher-v1' or payload.get('status') != 'completed'
            or payload.get('seed') != SEED or payload.get('purpose') != PURPOSE
            or len(payload.get('trajectories', [])) != 16):
        raise ValueError('Exactly sixteen completed Review10 teacher trajectories required')
    records = payload.get('records', [])
    receipts = {}
    all_forward_ids = set()
    for index, record in enumerate(records):
        if record.get('id') != index or record.get('status') != 'completed':
            raise ValueError('Every charged teacher candidate must have a verified complete result')
        receipt = deserialize_receipt(record['receipt'])
        evidence = record.get('physics_evidence', {})
        forward_ids = evidence.get('forward_calls')
        if (evidence.get('formal_receipt') is not True or evidence.get('status') != 'converged'
                or type(evidence.get('nsteps')) is not int or not 0 <= evidence['nsteps'] <= 300
                or evidence.get('raw_ef_attempts') != receipt.costs['raw_ef']
                or not 1 <= receipt.costs['raw_ef'] <= 301 or receipt.costs['relax_actual'] != 1):
            raise ValueError('Formal BFGS evidence and exact per-lane physical costs required')
        if (not isinstance(forward_ids,list) or len(forward_ids) != receipt.costs['raw_ef']
                or any(type(i) is not int or i < 1 for i in forward_ids)
                or forward_ids != sorted(set(forward_ids))):
            raise ValueError('Exact positive unique actual forward IDs are required for every physical lane')
        all_forward_ids.update(forward_ids)
        receipts[index] = receipt
    if len(records) != 240:
        raise ValueError('Fixed teacher design must account for all 240 initial/proposal evaluations')
    if (payload.get('counters', {}).get('relax_actual') != len(records)
            or payload['counters'].get('raw_ef') != sum(r.costs['raw_ef'] for r in receipts.values())
            or payload['counters']['raw_ef'] > 72240
            or payload['counters'].get('forward_calls') != len(all_forward_ids)
            or all_forward_ids != set(range(1,len(all_forward_ids)+1))):
        raise ValueError('Teacher ledger does not reconcile with actual physical receipts')
    rng = np.random.default_rng(seed_streams(SEED)[0])
    initial_seeds, initial_requests, seed_families = [], [], set()
    for _ in range(1000):
        seed = int(rng.integers(0,2**63-1))
        occupancy = env.validate_occupancy(make_state(seed,3,16).numbers)
        family = env.occupancy_key(occupancy)
        if family in seed_families: continue
        seed_families.add(family)
        initial_seeds.append(seed)
        initial_requests.append(env.build_request(occupancy).key)
        if len(initial_seeds) == 16: break
    if payload.get('initial_seeds') != initial_seeds:
        raise ValueError('Frozen teacher initial seed stream changed')
    consumed, initial_keys, validated = set(), set(), []
    for index, trajectory in enumerate(payload['trajectories']):
        if (trajectory.get('complete') is not True or trajectory.get('mode') != MIX[index]
                or trajectory.get('id') != index or len(trajectory.get('steps', [])) != 8):
            raise ValueError('Teacher mix and exactly eight attempted steps are fixed')
        initial_id = trajectory['initial_id']
        if initial_id in consumed or initial_id not in receipts: raise ValueError('Initial receipt reused or missing')
        consumed.add(initial_id)
        current_id = initial_id
        current = receipts[current_id]
        if payload.get('diagnostic') is False and current.request_key != initial_requests[index]:
            raise ValueError('Teacher initial state differs from frozen make_state seed')
        if trajectory.get('seed') != int(rng.integers(0,2**63-1)):
            raise ValueError('Teacher trajectory seed stream changed')
        initial_family = env.occupancy_key(current._request.occupancy)
        if initial_family in initial_keys: raise ValueError('Initial canonical families must be distinct')
        initial_keys.add(initial_family)
        families, rows, history = {current.terminal_key}, [], []
        best_energy = current.energy
        previous_breeding = None
        visited_forward_ids = set(records[initial_id]['physics_evidence']['forward_calls'])
        counts = env.Counters(relax_actual=1, raw_ef=current.costs['raw_ef'], forward_calls=len(visited_forward_ids))
        for step_index, step in enumerate(trajectory['steps']):
            ids = step.get('candidate_ids', [])
            if (step.get('step') != step_index or step.get('before_id') != current_id
                    or len(ids) != (4 if trajectory['mode'] == 'GA' else 1)
                    or len(set(ids)) != len(ids)):
                raise ValueError('Teacher timeline or candidate denominator changed')
            state = _state(current, counts, history, step_index, families, trajectory['seed'], best_energy)
            legal = env.enumerate_legal_actions(state.occupancy)
            if trajectory['mode'] == 'GA':
                expected_actions = ([json.loads(legal[int(i)].to_json()) for i in rng.choice(len(legal),4,replace=False)]
                                    if previous_breeding is None else [c['action'] for c in previous_breeding])
            else: expected_actions = [json.loads(legal[int(rng.integers(len(legal)))].to_json())]
            if [records[i]['action'] for i in ids] != expected_actions:
                raise ValueError('Teacher proposals differ from frozen uniform/genetic seed stream')
            for candidate_id in ids:
                if candidate_id in consumed or candidate_id not in receipts: raise ValueError('Candidate reused or missing')
                consumed.add(candidate_id)
                candidate, record = receipts[candidate_id], records[candidate_id]
                action = env.parse_action(json.dumps(record['action'], separators=(',', ':')))
                expected = env.build_request(env.apply_action(state.occupancy, action))
                if record.get('before_id') != current_id or expected.key != candidate.request_key:
                    raise ValueError('Teacher action does not connect to actual previous accepted terminal')
                families.add(candidate.terminal_key)
            accepted = step.get('accepted_id')
            if accepted is not None and accepted not in ids: raise ValueError('Accepted candidate outside step')
            if trajectory['mode'] in ('GA', 'random') and accepted is None: raise ValueError('Required transition omitted')
            if trajectory['mode'] == 'GA':
                ranked = sorted(ids, key=lambda i:(receipts[i].energy, i))
                if accepted != ranked[0] or step.get('selected_parent_ids') != ranked[:2]:
                    raise ValueError('GA selection must use actual relaxed energy')
                if previous_breeding is not None and [records[i]['action'] for i in ids] != [c['action'] for c in previous_breeding]:
                    raise ValueError('GA next generation does not use bred children')
                children = step.get('breeding', [])
                if len(children) != 4: raise ValueError('GA requires four bred action genomes')
                next_occupancy = records[accepted]['receipt']['terminal_occupancy']
                for child in children:
                    if (len(child.get('parent_ids', [])) != 2 or set(child['parent_ids']) != set(ranked[:2])
                            or child.get('mutated') is not True or child.get('mutation_endpoint') not in (0,1)):
                        raise ValueError('GA child lacks measured parents and endpoint mutation')
                    parents = [env.parse_action(json.dumps(records[i]['action'])) for i in child['parent_ids']]
                    crossed, mutated = child.get('crossover_sites', []), child.get('mutated_sites', [])
                    endpoint = child['mutation_endpoint']
                    if (len(crossed) != 2 or crossed[0] not in parents[0].sites or crossed[1] not in parents[1].sites
                            or len(mutated) != 2 or mutated[1-endpoint] != crossed[1-endpoint]):
                        raise ValueError('GA crossover or mutation evidence changed')
                    env.apply_action(next_occupancy, env.parse_action(json.dumps(child['action'])))
                from smoke_teacher10 import breed_population
                expected_children = breed_population(next_occupancy,[records[i] for i in ranked[:2]],rng)
                if children != expected_children: raise ValueError('GA breeding differs from frozen crossover/mutation/repair')
                previous_breeding = children
            if trajectory['mode'] == 'SA':
                temperature = float(np.linspace(.20,.02,8)[step_index])
                uniform = step.get('acceptance_uniform')
                expected_uniform = float(rng.random())
                delta = receipts[ids[0]].energy-current.energy
                if (step.get('temperature_eV') != temperature or type(uniform) is not float or uniform != expected_uniform
                        or (accepted is not None) != (delta <= 0 or uniform < np.exp(-delta/temperature))):
                    raise ValueError('SA acceptance contradicts fixed Metropolis decision')
            for candidate_id in ids:
                if records[candidate_id].get('accepted') is not (candidate_id == accepted):
                    raise ValueError('Candidate acceptance flag contradicts timeline')
            chosen_id = ids[0] if accepted is None else accepted
            chosen = receipts[chosen_id]
            record = records[chosen_id]
            if accepted is not None:
                rows.append({'messages':env.encode_prompt_v2(state), 'action':record['action'],
                    'physical_receipt':chosen.summary(), 'trajectory_id':index, 'step':step_index,
                    'before_receipt_id':current_id, 'after_receipt_id':accepted})
            history.append({'action':env.parse_action(json.dumps(record['action'])).to_json(),
                'status':'success' if accepted is not None else 'rejected', 'delta':chosen.energy-current.energy,
                'accepted_energy':chosen.energy if accepted is not None else None})
            counts.attempts += 1
            counts.relax_actual += len(ids)
            counts.raw_ef += sum(receipts[i].costs['raw_ef'] for i in ids)
            visited_forward_ids.update(forward_id for i in ids for forward_id in records[i]['physics_evidence']['forward_calls'])
            counts.forward_calls = len(visited_forward_ids)
            if accepted is not None:
                current_id, current = accepted, chosen
                best_energy = min(best_energy, current.energy)
        if trajectory.get('terminal_id') != current_id: raise ValueError('Wrong terminal reference')
        validated.append({'id':index, 'rows':rows, 'terminal_families':sorted(families), 'complete':True})
    if consumed != set(receipts): raise ValueError('Unattached physical candidates')
    if payload['counters'].get('attempts') != 128: raise ValueError('Exactly 128 teacher decisions required')
    return validated


def distinct_unscored_rows(count, excluded_families, *, rng, with_actions=False):
    """CPU-only unique canonical states; balanced operations, uniform within each op."""
    used = set(excluded_families)
    rows = []
    for _ in range(max(1000, count * 20)):
        seed = int(rng.integers(0, 2**63-1))
        occupancy = env.validate_occupancy(make_state(seed, 3, 16).numbers)
        key = env.occupancy_key(occupancy)
        if key in used: continue
        used.add(key)
        row = {'seed':seed, 'occupancy':list(occupancy), 'canonical_key':key, 'unscored':True,
               'energy_eV':None, 'physical_calls':0, 'DFT_labels_used':False,
               'messages':env.encode_prompt_v2(unscored_state(occupancy))}
        if with_actions:
            op = ('swap_cations', 'move_vacancy')[len(rows) % 2]
            legal = [a for a in env.enumerate_legal_actions(occupancy) if a.op == op]
            row['action'] = json.loads(legal[int(rng.integers(len(legal)))].to_json())
        rows.append(row)
        if len(rows) == count: return rows
    raise ValueError('Could not construct requested distinct canonical families')


def freeze_data(teacher_result, output, *, excluded_families):
    """Explicit post-recovery freeze, never invoked by import or by teacher collection."""
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    payload = json.loads(Path(teacher_result).read_text()) if not isinstance(teacher_result, dict) else teacher_result
    if (payload.get('runtime', {}).get('GPU_work_performed') is not True
            or payload.get('diagnostic') is not False):
        raise ValueError('CPU synthetic diagnostics cannot become physical teacher training data')
    trajectories = verify_teacher_result(payload)
    sources = source_hashes()
    if payload.get('source_hashes') != sources or sources['claude10_physics.py'] != PHYSICS_SOURCE_SHA:
        raise ValueError('Frozen teacher/source fingerprints changed')
    excluded = set(excluded_families)
    if not excluded: raise ValueError('Existing 633/650 heldout families must be explicitly supplied')
    all_teacher = set().union(*(set(t['terminal_families']) for t in trajectories))
    if all_teacher & excluded: raise ValueError('Teacher families overlap existing heldout exclusions')
    train, valid = split_terminal_families(trajectories, seed=seed_streams(SEED)[1])
    rng = np.random.default_rng(seed_streams(SEED)[2])
    forbidden = excluded | all_teacher
    files = {}
    files['format_train.jsonl'] = distinct_unscored_rows(4096, forbidden, rng=rng, with_actions=True)
    forbidden.update(r['canonical_key'] for r in files['format_train.jsonl'])
    for name, count in [('legality_sft.jsonl',80), ('legality_grpo.jsonl',20), ('starts.jsonl',4)]:
        files[name] = distinct_unscored_rows(count, forbidden, rng=rng)
        forbidden.update(r['canonical_key'] for r in files[name])
    files['teacher_train.jsonl'] = [r for t in train for r in t['rows']]
    files['teacher_validation.jsonl'] = [r for t in valid for r in t['rows']]
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in files.items():
        (output / name).write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False)+'\n' for r in rows))
    write_json(output / 'teacher_result.json', payload)
    manifest = {'schema':'review10-data-v1', 'seed':SEED, 'purpose':PURPOSE,
        'source_hashes':sources, 'counts':{name:len(rows) for name,rows in files.items()},
        'teacher_train_trajectory_ids':[t['id'] for t in train],
        'teacher_validation_trajectory_ids':[t['id'] for t in valid],
        'excluded_families':sorted(excluded), 'format_physical_calls':0, 'DFT_labels_used':False,
        'teacher_oversample':8, 'teacher_oversample_interpretation':'repeated_examples_not_independent_data',
        'files':{name:sha256(output/name) for name in list(files) + ['teacher_result.json']}}
    write_json(output / 'manifest.json', manifest)
    return manifest


def load_frozen_data(path):
    """Hash and semantic recovery before a tokenizer/model can consume the freeze."""
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    if (manifest.get('schema') != 'review10-data-v1' or manifest.get('seed') != SEED
            or manifest.get('purpose') != PURPOSE or manifest.get('source_hashes') != source_hashes()
            or manifest.get('teacher_oversample') != 8):
        raise ValueError('Review10 frozen data/source binding changed')
    names = {'format_train.jsonl', 'teacher_train.jsonl', 'teacher_validation.jsonl',
             'legality_sft.jsonl', 'legality_grpo.jsonl', 'starts.jsonl', 'teacher_result.json'}
    if set(manifest.get('files', {})) != names: raise ValueError('Frozen data file set changed')
    for name, digest in manifest['files'].items():
        if sha256(path/name) != digest: raise ValueError('Frozen data hash changed: '+name)
    trajectories = verify_teacher_result(json.loads((path/'teacher_result.json').read_text()))
    split = split_terminal_families(trajectories, seed=seed_streams(SEED)[1])
    rows = {name:[json.loads(line) for line in (path/name).read_text().splitlines()] for name in names if name.endswith('.jsonl')}
    for name, group in zip(['teacher_train.jsonl','teacher_validation.jsonl'], split):
        if rows[name] != [r for t in group for r in t['rows']]: raise ValueError('Teacher rows differ from recovered physical timeline')
    used = set(manifest['excluded_families']) | set().union(*(set(t['terminal_families']) for t in trajectories))
    for name, count in [('format_train.jsonl',4096), ('legality_sft.jsonl',80), ('legality_grpo.jsonl',20), ('starts.jsonl',4)]:
        if len(rows[name]) != count: raise ValueError('Frozen count changed: '+name)
        for row in rows[name]:
            occupancy = env.validate_occupancy(row['occupancy'])
            key = env.occupancy_key(occupancy)
            if (row['canonical_key'] != key or key in used or row.get('unscored') is not True
                    or row.get('energy_eV') is not None or row.get('physical_calls') != 0
                    or row['messages'] != env.encode_prompt_v2(unscored_state(occupancy))):
                raise ValueError('Unscored family, observation, or metadata changed')
            used.add(key)
            if name == 'format_train.jsonl': env.apply_action(occupancy, env.parse_action(json.dumps(row['action'])))
    if sum(r['action']['op']=='swap_cations' for r in rows['format_train.jsonl']) != 2048:
        raise ValueError('Format operation balance changed')
    return {'manifest':manifest, 'rows':rows, 'manifest_sha256':sha256(path/'manifest.json')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--excluded-families', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    exclusions = json.loads(args.excluded_families.read_text())
    if not isinstance(exclusions, list) or not all(isinstance(k,str) for k in exclusions):
        raise ValueError('Excluded families file must be a JSON list of canonical keys')
    print(json.dumps(freeze_data(args.teacher, args.output, excluded_families=exclusions)['counts']))


if __name__ == '__main__': main()
