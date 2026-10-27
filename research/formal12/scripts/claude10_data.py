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
PHYSICS_SOURCE_SHA = '47a726cef4e8ad11b4c2d66a474795ed4b4a0cfd6e71851220fd3507b7c48730'
PROTOCOL_SHA = env.protocol_fingerprint(env.RelaxationSpec())
TEACHER_SCHEMA = 'review10-v22-teacher-v1'
DATA_SCHEMA = 'review10-v22-data-v1'
SFT_SCHEMA = 'review10-v22-sft-v1'


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name:sha256(root / name) for name in
            ['claude08_env.py', 'claude08_data.py', 'claude10_physics.py',
             'claude10_data.py', 'smoke_teacher10.py', 'smoke_sft10.py',
             'smoke_mace.py', 'calibration.py', 'smoke_grpo08.py']}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def persist_result(path, result, *, started_at, clock=time.monotonic, wall_seconds=6600):
    """Check after both result writes so a late write cannot leave a released status."""
    before = clock()
    result['wall_seconds'] = before-started_at
    write_json(path,result)
    after = clock()
    result.update(wall_seconds=after-started_at, result_persist_seconds=after-before)
    overrun = after-started_at >= wall_seconds
    if overrun:
        result.update(status='failed',error='Entire-stage runtime cap reached during cleanup/persist',
                      adapter_release_allowed=False)
    write_json(path,result)
    final = clock()
    if final-started_at >= wall_seconds:
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
    return {'protocol':env.PROTOCOL, 'protocol_sha':receipt._request.protocol_sha,
            'perturbation_seed':receipt._request.perturbation_seed,
            'request_occupancy':list(receipt._request.occupancy),
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
        if (request.spec != env.RelaxationSpec() or payload.get('protocol') != env.PROTOCOL
                or payload.get('protocol_sha') != request.protocol_sha
                or payload.get('perturbation_seed') != request.perturbation_seed):
            raise ValueError('Serialized receipt is not bound to the complete v2.2 protocol')
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


def deserialize_result(payload):
    return env.PhysicsResult(Atoms(numbers=payload['numbers'],positions=payload['positions'],
            cell=payload['cell'],pbc=payload['pbc']),payload['energy_eV'],
            np.asarray(payload['forces'],dtype=float),payload['converged'])


def recover_teacher_record(record):
    """Recompute force convergence and strict mapping, including claimed rejections."""
    try:
        request=env.build_request(record['request_occupancy'],env.RelaxationSpec(**record['spec']))
        if (request.spec != env.RelaxationSpec() or record.get('protocol') != env.PROTOCOL
                or record.get('protocol_sha') != request.protocol_sha
                or record.get('perturbation_seed') != request.perturbation_seed
                or record.get('request_key') != request.key):
            raise ValueError('Teacher request is not bound to the complete v2.2 protocol')
        costs=record['costs']; evidence=record['physics_evidence']
        ids=evidence.get('forward_calls')
        if (type(costs.get('relax_actual')) is not int or costs['relax_actual'] != 1
                or type(costs.get('raw_ef')) is not int or not 1 <= costs['raw_ef'] <= 301
                or evidence.get('formal_receipt') is not True or evidence.get('status') != 'converged'
                or type(evidence.get('nsteps')) is not int or not 0 <= evidence['nsteps'] <= 300
                or evidence.get('raw_ef_attempts') != costs['raw_ef']
                or evidence.get('protocol') != env.PROTOCOL or evidence.get('protocol_sha') != request.protocol_sha
                or evidence.get('perturbation_seed') != request.perturbation_seed
                or evidence.get('relaxation_spec') != asdict(request.spec)):
            raise ValueError('Full converged v2.2 LBFGS evidence and exact physical costs required')
        if (not isinstance(ids,list) or len(ids) != costs['raw_ef']
                or any(type(i) is not int or i < 1 for i in ids) or ids != sorted(set(ids))):
            raise ValueError('Exact positive unique actual forward IDs are required for every physical lane')
        raw=deserialize_result(record['result'])
        try:
            terminal=env.verify_result(raw,request.spec)
        except env.UnmappableTerminal:
            terminal=None
        diagnostics=env.mapping_diagnostics(raw.atoms)
        if record.get('mapping_diagnostics') != diagnostics:
            raise ValueError('Actual geometry mapping diagnostics differ from claimed metadata')
        receipt=None
        if terminal is None:
            if (record.get('status') != 'UNMAPPABLE_TERMINAL' or 'receipt' in record
                    or record.get('accepted') is not False
                    or any(record.get(k) != 0. for k in ('energy_reward','novelty_reward','reward'))):
                raise ValueError('Unmappable geometry must remain a rejected zero-reward result without a receipt')
        else:
            if record.get('status') != 'completed': raise ValueError('Claimed unmappable result actually maps')
            receipt=deserialize_receipt(record['receipt'])
            if (receipt.request_key != request.key or dict(receipt.costs) != costs
                    or record['receipt']['result'] != record['result']):
                raise ValueError('Raw result and independently verified receipt do not agree')
        return {'request':request,'result':raw,'terminal':terminal,'receipt':receipt,
                'mapping_diagnostics':diagnostics,'forward_ids':tuple(ids),'costs':costs}
    except (KeyError,TypeError,AttributeError) as exc:
        raise ValueError('Complete actual v2.2 teacher result required') from exc


def verify_teacher_result(payload):
    """Recover every charged mapped/unmapped result, then replay the exact teacher."""
    if (payload.get('schema') != TEACHER_SCHEMA or payload.get('status') != 'completed'
            or payload.get('protocol') != env.PROTOCOL or payload.get('protocol_sha') != PROTOCOL_SHA
            or payload.get('checkpoint_sha256') != env.MODEL_SHA or payload.get('teacher_rounds') != 128
            or payload.get('seed') != SEED or payload.get('purpose') != PURPOSE
            or len(payload.get('trajectories',[])) != 16):
        raise ValueError('Exactly sixteen completed v2.2 teacher trajectories required')
    records=payload.get('records',[])
    if len(records) != 240: raise ValueError('All 240 charged initial/proposal records are required')
    recovered={}
    for index,record in enumerate(records):
        if record.get('id') != index: raise ValueError('Teacher record IDs changed')
        recovered[index]=recover_teacher_record(record)
    attempts=unmapped=0
    all_forward_ids=set()
    offset=batch_id=0
    while offset < len(records):
        batch_id+=1
        size=records[offset].get('batch_size')
        if type(size) is not int or not 1 <= size <= 4: raise ValueError('Invalid physical batch size')
        group=records[offset:offset+size]
        initial=offset < 16
        if not initial: attempts+=size
        for lane,record in enumerate(group):
            if (record.get('batch_id') != batch_id or record.get('lane') != lane
                    or record.get('batch_size') != size or record.get('is_initial') is not initial
                    or record.get('attempts_charged') != attempts
                    or record.get('physical_action_attempts_charged') != attempts):
                raise ValueError('Physical attempts must be charged once per candidate before each batch')
            item=recovered[record['id']]
            all_forward_ids.update(item['forward_ids'])
            if item['terminal'] is None:
                if initial: raise ValueError('An unmappable initial state is a hard failure')
                unmapped+=1
            if not initial and env.unmappable_rate_exceeded(attempts, unmapped):
                raise ValueError('Completed teacher crossed the strict ten-percent stop after 40 physical actions')
        offset+=size
    counters=payload.get('counters',{})
    if (attempts != 224 or counters.get('attempts') != attempts or counters.get('physical_action_attempts') != attempts
            or counters.get('relax_actual') != 240
            or counters.get('unmappable_terminals') != unmapped
            or counters.get('raw_ef') != sum(item['costs']['raw_ef'] for item in recovered.values())
            or counters['raw_ef'] > 72240 or counters.get('forward_calls') != len(all_forward_ids)
            or all_forward_ids != set(range(1,len(all_forward_ids)+1))):
        raise ValueError('Teacher actual-cost ledger does not reconcile')
    if payload.get('near_threshold_count') != sum(item['mapping_diagnostics']['NEAR_THRESHOLD'] for item in recovered.values()):
        raise ValueError('Teacher near-threshold diagnostic count changed')
    for name, flag in [('ambiguous_count','AMBIGUOUS'),('occupation_tie_count','TIE')]:
        if payload.get(name) != sum(item['mapping_diagnostics'][flag] for item in recovered.values()):
            raise ValueError('Teacher '+name+' diagnostic count changed')
    rng=np.random.default_rng(seed_streams(SEED)[0])
    initial_seeds=[]; initial_requests=[]; seed_families=set()
    for _ in range(1000):
        seed=int(rng.integers(0,2**63-1))
        occupancy=env.validate_occupancy(make_state(seed,3,16).numbers)
        family=env.occupancy_key(occupancy)
        if family in seed_families: continue
        seed_families.add(family); initial_seeds.append(seed); initial_requests.append(env.build_request(occupancy).key)
        if len(initial_seeds)==16: break
    if payload.get('initial_seeds') != initial_seeds: raise ValueError('Frozen initial seed stream changed')
    validated=[]; consumed=set(); initial_keys=set(); novelty_events=near_novelty_events=0
    for index,trajectory in enumerate(payload['trajectories']):
        if (trajectory.get('complete') is not True or trajectory.get('mode') != MIX[index]
                or trajectory.get('id') != index or len(trajectory.get('steps',[])) != 8):
            raise ValueError('Teacher mix and eight decision rounds are fixed')
        initial_id=trajectory['initial_id']
        if initial_id != index or initial_id in consumed: raise ValueError('Initial receipt reused or missing')
        consumed.add(initial_id)
        current_id=initial_id; current=recovered[current_id]['receipt']
        if current is None: raise ValueError('Mapped initial receipt required')
        if payload.get('diagnostic') is False and current.request_key != initial_requests[index]:
            raise ValueError('Teacher initial state differs from frozen make_state seed')
        if trajectory.get('seed') != int(rng.integers(0,2**63-1)): raise ValueError('Trajectory seed stream changed')
        family=env.occupancy_key(current._request.occupancy)
        if family in initial_keys: raise ValueError('Initial canonical families must be distinct')
        initial_keys.add(family)
        mapped_families={current.terminal_key}; request_families={family}; visits={current.terminal_key}
        best=current.energy; rows=[]; history=[]; population=None
        forward_ids=set(recovered[initial_id]['forward_ids'])
        counts=env.Counters(relax_actual=1,raw_ef=current.costs['raw_ef'],forward_calls=len(forward_ids))
        for step_index,step in enumerate(trajectory['steps']):
            ids=step.get('candidate_ids',[])
            mode=trajectory['mode']
            if (step.get('step') != step_index or step.get('before_id') != current_id
                    or len(ids) != (4 if mode=='GA' else 1) or len(set(ids)) != len(ids)):
                raise ValueError('Teacher timeline or candidate denominator changed')
            state=_state(current,counts,history,step_index,visits,trajectory['seed'],best)
            legal=env.enumerate_legal_actions(state.occupancy)
            if mode=='GA':
                if population is None:
                    population=[json.loads(legal[int(i)].to_json()) for i in rng.choice(len(legal),4,replace=False)]
                expected_actions=population
            else: expected_actions=[json.loads(legal[int(rng.integers(len(legal)))].to_json())]
            if [records[i]['action'] for i in ids] != expected_actions:
                raise ValueError('Teacher proposals differ from frozen uniform/genetic seed stream')
            for candidate_id in ids:
                if candidate_id in consumed or candidate_id not in recovered: raise ValueError('Candidate reused or missing')
                consumed.add(candidate_id)
                item=recovered[candidate_id]; record=records[candidate_id]
                expected=env.build_request(env.apply_action(state.occupancy,env.parse_action(json.dumps(record['action']))))
                if record.get('before_id') != current_id or item['request'].key != expected.key:
                    raise ValueError('Teacher action does not connect to actual previous accepted terminal')
                request_families.add(env.occupancy_key(item['request'].occupancy))
                if item['terminal'] is not None: mapped_families.add(item['terminal'].mapping.canonical_key)
            mapped=[i for i in ids if recovered[i]['receipt'] is not None]
            ranked=sorted(mapped,key=lambda i:(recovered[i]['receipt'].energy,i))
            if step.get('mapped_candidate_ids') != mapped: raise ValueError('Mapped candidate list changed')
            accepted=None; delta=None
            if mode=='SA':
                uniform=float(rng.random()); temperature=float(np.linspace(.20,.02,8)[step_index])
                if mapped:
                    delta=recovered[ids[0]]['receipt'].energy-current.energy
                    if delta <= 0 or uniform < np.exp(-delta/temperature): accepted=ids[0]
                if (step.get('temperature_eV') != temperature or step.get('acceptance_uniform') != uniform):
                    raise ValueError('SA fixed Metropolis stream changed')
            elif mode=='random':
                if mapped: accepted=ids[0]; delta=recovered[accepted]['receipt'].energy-current.energy
            elif len(ranked)>=2:
                accepted=ranked[0]; delta=recovered[accepted]['receipt'].energy-current.energy
                from smoke_teacher10 import breed_population
                children=breed_population(records[accepted]['receipt']['terminal_occupancy'],
                                           [records[i] for i in ranked[:2]],rng)
                if (step.get('selected_parent_ids') != ranked[:2] or step.get('breeding') != children
                        or step.get('fallback') is not None):
                    raise ValueError('GA selection/breeding differs from mapped measured fitness')
                population=[child['action'] for child in children]
            else:
                if (step.get('selected_parent_ids') != [] or step.get('breeding') != []
                        or step.get('fallback') != 'INSUFFICIENT_MAPPED_KEEP_POPULATION'):
                    raise ValueError('GA insufficient-mapped fallback must preserve the original population')
            if step.get('accepted_id') != accepted or step.get('energy_delta_eV') != delta:
                raise ValueError('Acceptance or actual energy delta contradicts replay')
            chosen_id=ids[0] if accepted is None else accepted
            chosen=recovered[chosen_id]['receipt']
            energy_reward=novelty_reward=0.
            if accepted is not None:
                energy_reward=max(0.,best-min(best,chosen.energy))/.20
                novelty_reward=.05 if chosen.terminal_key not in visits else 0.
            for candidate_id in ids:
                record=records[candidate_id]
                expected_energy=energy_reward if candidate_id==accepted else 0.
                expected_novelty=novelty_reward if candidate_id==accepted else 0.
                if (record.get('accepted') is not (candidate_id==accepted)
                        or record.get('energy_reward') != expected_energy or record.get('novelty_reward') != expected_novelty
                        or record.get('reward') != expected_energy+expected_novelty):
                    raise ValueError('Candidate acceptance/reward contradicts the mapped transition')
            if accepted is not None:
                rows.append({'messages':env.encode_prompt_v2(state),'action':records[accepted]['action'],
                    'physical_receipt':chosen.summary(),'trajectory_id':index,'step':step_index,
                    'before_receipt_id':current_id,'after_receipt_id':accepted})
            status=('success' if accepted is not None else 'GA_INSUFFICIENT_MAPPED' if mode=='GA'
                    else 'UNMAPPABLE_TERMINAL' if not mapped else 'rejected')
            history.append({'action':env.parse_action(json.dumps(records[chosen_id]['action'])).to_json(),
                            'status':status,'delta':delta})
            counts.attempts+=len(ids); counts.physical_action_attempts+=len(ids); counts.relax_actual+=len(ids)
            counts.raw_ef+=sum(recovered[i]['costs']['raw_ef'] for i in ids)
            counts.unmappable_terminals+=len(ids)-len(mapped)
            forward_ids.update(forward for i in ids for forward in recovered[i]['forward_ids'])
            counts.forward_calls=len(forward_ids)
            if accepted is not None:
                if chosen.terminal_key not in visits:
                    counts.novelty_events+=1; novelty_events+=1
                    if recovered[accepted]['mapping_diagnostics']['NEAR_THRESHOLD']:
                        counts.near_threshold_novelty_events+=1; near_novelty_events+=1
                visits.add(chosen.terminal_key); current_id=accepted; current=chosen; best=min(best,current.energy)
        if (trajectory.get('terminal_id') != current_id or trajectory.get('best_energy_eV') != best
                or trajectory.get('private_visits') != sorted(visits)):
            raise ValueError('Terminal/best/visits differ from accepted mapped-state replay')
        validated.append({'id':index,'rows':rows,'terminal_families':sorted(mapped_families),
            'request_families':sorted(request_families),'exclusion_families':sorted(mapped_families|request_families),'complete':True})
    if (consumed != set(recovered) or counters.get('novelty_events') != novelty_events
            or counters.get('near_threshold_novelty_events') != near_novelty_events
            or payload.get('near_novelty_fraction') != (near_novelty_events/novelty_events if novelty_events else None)):
        raise ValueError('Unattached physical candidates or novelty ledger mismatch')
    return validated


def split_teacher_families(trajectories):
    """Keep overlap through requests and mapped terminals in a single component."""
    by_id={t['id']:t for t in trajectories}
    groups=split_terminal_families([{'id':t['id'],'terminal_families':t['exclusion_families']} for t in trajectories],
                                  seed=seed_streams(SEED)[1])
    return tuple([by_id[t['id']] for t in group] for group in groups)


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
    all_teacher = set().union(*(set(t['exclusion_families']) for t in trajectories))
    if all_teacher & excluded: raise ValueError('Teacher families overlap existing heldout exclusions')
    train, valid = split_teacher_families(trajectories)
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
    manifest = {'schema':DATA_SCHEMA, 'protocol':env.PROTOCOL,'protocol_sha':PROTOCOL_SHA,
        'checkpoint_sha256':env.MODEL_SHA, 'seed':SEED, 'purpose':PURPOSE,
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
    if (manifest.get('schema') != DATA_SCHEMA or manifest.get('seed') != SEED
            or manifest.get('protocol') != env.PROTOCOL or manifest.get('protocol_sha') != PROTOCOL_SHA
            or manifest.get('checkpoint_sha256') != env.MODEL_SHA
            or manifest.get('purpose') != PURPOSE or manifest.get('source_hashes') != source_hashes()
            or manifest.get('teacher_oversample') != 8):
        raise ValueError('Review10 frozen data/source binding changed')
    names = {'format_train.jsonl', 'teacher_train.jsonl', 'teacher_validation.jsonl',
             'legality_sft.jsonl', 'legality_grpo.jsonl', 'starts.jsonl', 'teacher_result.json'}
    if set(manifest.get('files', {})) != names: raise ValueError('Frozen data file set changed')
    for name, digest in manifest['files'].items():
        if sha256(path/name) != digest: raise ValueError('Frozen data hash changed: '+name)
    trajectories = verify_teacher_result(json.loads((path/'teacher_result.json').read_text()))
    split = split_teacher_families(trajectories)
    rows = {name:[json.loads(line) for line in (path/name).read_text().splitlines()] for name in names if name.endswith('.jsonl')}
    for name, group in zip(['teacher_train.jsonl','teacher_validation.jsonl'], split):
        if rows[name] != [r for t in group for r in t['rows']]: raise ValueError('Teacher rows differ from recovered physical timeline')
    used = set(manifest['excluded_families']) | set().union(*(set(t['exclusion_families']) for t in trajectories))
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
