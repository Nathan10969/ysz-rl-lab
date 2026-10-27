"""Review10 synchronous G4/K8 runner; importing never loads a model or starts work.

All four actions at a rollout step are generated and reference-scored on the
main thread. The unchanged Review08 Environment.step owns attempts, request
cache, reward and private novelty. Serialized workers suspend only its physical
callback; the main thread executes the resulting 1--4 real requests together.
No pending request is deduplicated or treated as a known cache entry.

This module has no submission, input-freeze or overlay command. The real CLI is
an in-job entry point and requires a hash-bound Review10 release first.
"""
from __future__ import annotations

import time
PROCESS_STARTED_AT = time.monotonic()
import argparse
from concurrent.futures import Future
from dataclasses import asdict
import json
import math
from pathlib import Path
import threading
import traceback

import numpy as np

import claude08_env as env
from claude08_data import sha256, unscored_state
from claude08_grpo import group_advantages
from claude10_physics import BatchMACEPhysics
import smoke_grpo08 as old
from smoke_grpo08 import (SEED, GROUPS, GROUP_SIZE, ATTEMPTS, LIMITS,
    INTERNAL_SECONDS, FUTURE_OUTER_SECONDS, FUTURE_SLURM_SECONDS,
    HFActor, TokenLedger, GeneratedAction, track_phase, save_json)


def source_hashes():
    """Bind the runner and all project sources used by release/execution."""
    root = Path(__file__).resolve().parent
    names = ('smoke_grpo10.py', 'smoke_grpo08.py', 'claude08_grpo.py',
             'claude08_env.py', 'claude08_data.py', 'claude10_physics.py',
             'claude10_data.py', 'smoke_teacher10.py', 'smoke_sft10.py',
             'smoke_mace.py', 'smoke_common.py', 'calibration.py', 'lattice.py')
    return {name:sha256(root/name) for name in names}


class _StepBarrier:
    """One synchronous step, with all Environment mutations behind one RLock.

    Each worker enters once in deterministic lane order. A callback announces
    its already precharged request and waits while releasing the same lock.
    Main-thread physics resolves lanes in order, allowing old step to verify,
    insert cache and update its state before the next worker resumes.
    """
    def __init__(self, environments, ledger, execute_batch):
        self.environments, self.ledger, self.execute_batch = environments, ledger, execute_batch
        self.condition = threading.Condition(threading.RLock())
        self.slots = [dict(ready=False, done=False, future=None, request=None,
                           outcome=None, error=None) for _ in environments]
        self.threads = []
        for lane, environment in enumerate(environments):
            environment.callback = self._callback(lane)

    def _remaining(self):
        self.ledger.raw.check_time()
        remaining = self.ledger.limits.wall_seconds - (self.ledger.clock()-self.ledger.started_at)
        return min(.1, max(.001, remaining))

    def _wait(self, predicate):
        while not predicate():
            self.condition.wait(timeout=self._remaining())

    def _callback(self, lane):
        def callback(request, counter):
            # Called while the worker owns condition's RLock exactly once.
            slot = self.slots[lane]
            slot.update(ready=True, request=request, future=Future())
            self.condition.notify_all()
            self._wait(slot['future'].done)
            return slot['future'].result()
        return callback

    def _worker(self, lane, text):
        with self.condition:
            slot = self.slots[lane]
            try: slot['outcome'] = self.environments[lane].step(text)
            except BaseException as exc: slot['error'] = exc
            finally:
                slot['done'] = True
                self.condition.notify_all()

    def _resolve(self, lane, *, value=None, error=None):
        with self.condition:
            slot = self.slots[lane]
            future = slot['future']
            if future is not None and not future.done():
                if error is None: future.set_result(value)
                else: future.set_exception(error)
                self.condition.notify_all()
            self._wait(lambda:slot['done'])

    def run(self, actions):
        failure = None
        try:
            for lane, action in enumerate(actions):
                thread = threading.Thread(target=self._worker,args=(lane,action.text),
                    name='review10-step-'+str(lane),daemon=True)
                self.threads.append(thread)
                thread.start()
                with self.condition:
                    self._wait(lambda:self.slots[lane]['ready'] or self.slots[lane]['done'])
            pending = [i for i,s in enumerate(self.slots) if s['future'] is not None]
            if pending:
                try:
                    values = self.execute_batch([self.slots[i]['request'] for i in pending])
                    if len(values) != len(pending): raise RuntimeError('Physical result count differs from real lanes')
                except BaseException as exc:
                    failure = exc
                    values = [None]*len(pending)
                for lane,value in zip(pending,values):
                    self._resolve(lane,value=value,error=failure)
            errors = [s['error'] for s in self.slots if s['error'] is not None]
            if failure is None and errors: failure = errors[0]
        except BaseException as exc:
            failure = exc
        finally:
            # Resolve every suspended callback even on main/worker/time errors.
            with self.condition:
                for slot in self.slots:
                    future = slot['future']
                    if future is not None and not future.done():
                        future.set_exception(failure or RuntimeError('Synchronous physical barrier aborted'))
                self.condition.notify_all()
            for thread in self.threads:
                # Workers have no GPU/HF calls; this also bounds error cleanup.
                remaining = self.ledger.limits.wall_seconds-(self.ledger.clock()-self.ledger.started_at)
                thread.join(timeout=max(.01,min(1.,remaining)))
            if any(thread.is_alive() for thread in self.threads):
                failure = failure or RuntimeError('Step worker did not finish within stage deadline')
        return [s['outcome'] for s in self.slots], failure


def _validate_protocol(limits, starts, heldout):
    for name,value in asdict(limits).items():
        bound = getattr(LIMITS,name)
        if value is None or (bound is not None and value > bound):
            raise ValueError('CPU injected limits may only tighten the frozen protocol')
    if limits.max_attempts != ATTEMPTS: raise ValueError('Review10 always uses K8')
    if len(starts) != GROUPS or len(heldout) != 20:
        raise ValueError('Exactly four starts and twenty frozen legality states required')


def _terminal_evidence(terminal):
    result, mapping = terminal.result, terminal.mapping
    return dict(energy_eV=result.energy, forces_eV_A=result.forces.tolist(),
        converged=True, fmax_eV_A=float(np.linalg.norm(result.forces,axis=1).max()),
        mapped_occupancy=list(mapping.occupancy),canonical_key=mapping.canonical_key,
        geometry_hash=mapping.geometry_hash,assigned_site_ids=list(mapping.assigned_site_ids),
        assignment_distances_A=list(mapping.assignment_distances),shell_counts=mapping.shell_counts,
        positions_A=result.atoms.positions.tolist(),numbers=result.atoms.numbers.tolist(),
        cell_A=result.atoms.cell.array.tolist(),pbc=result.atoms.pbc.tolist())


def run_smoke(actor, physics, starts, heldout, *, output=None, limits=LIMITS,
              started_at=None, clock=time.monotonic, runtime_context=None):
    """Run the fixed gate/4-start/G4/K8 protocol with one eventual joint update.

    Actor is the unchanged HFActor contract. Physics exposes relax_many(requests,
    shared_raw_counter,max_steps=300), preserving result lane order and actual
    formal-BFGS evidence. Fake dependencies are for CPU tests only.
    """
    _validate_protocol(limits,starts,heldout)
    output = None if output is None else Path(output)
    if output is not None: output.mkdir(parents=True,exist_ok=False)
    counters = env.Counters()
    ledger = TokenLedger(counters,limits,started_at=started_at,clock=clock,
                        log_path=None if output is None else output/'cost_attempts.jsonl')
    ledger.raw.phase_ledger = ledger
    result = dict(schema='review10-grpo-v1',status='STARTED',seed=SEED,limits=asdict(limits),
        source_hashes=source_hashes(),legality=dict(denominator=20,legal=0,generations=[]),
        initial_relaxations=0,physical_batches=[],groups=[],partial_episodes=[],optimizer_updates=0,
        usable_for_workflow=False,scientific_efficacy_claim=False,runtime_versions=old.runtime_versions(),
        count_definition='relax_actual precharged by caller; raw E/F charged by actual model prehook per structure; full-context input prefill+seen; sampled action/EOS new+seen',
        backstops=dict(future_outer_seconds=FUTURE_OUTER_SECONDS,future_slurm_seconds=FUTURE_SLURM_SECONDS,
                       OS_IO_hard_preemption=False))
    if runtime_context is not None:
        if set(runtime_context)-{'runtime','release','data_manifest_sha256'}:
            raise ValueError('Unknown runtime evidence key')
        result.update(runtime_context)
    cache, history, initial_keys = env.RequestCache(), set(), set()
    episodes, returns = [], []
    phase = {'name':'initial'}

    def execute_batch(requests):
        """The only physical execution entry; caller has charged all attempts."""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Only main thread may execute physics')
        ledger.raw.check_time()
        if not 1 <= len(requests) <= 4: raise ValueError('One to four real requests required')
        before = asdict(counters)
        batch_id = len(result['physical_batches'])
        record = dict(batch_id=batch_id,phase=dict(phase),status='started',batch_size=len(requests),
                      begin_elapsed_seconds=clock()-ledger.started_at,counts_before=before,lanes=[],
                      warmup_raw_ef=0,raw_ef_definition='actual evaluated structures; no warmup calls; failed forwards retained')
        filename = 'physics/batch_'+str(batch_id).zfill(4)+'.json'
        record['receipt_file'] = filename if output is not None else None
        for lane,request in enumerate(requests):
            record['lanes'].append(dict(lane=lane,request_key=request.key,status='started',
                protocol=env.PROTOCOL,relaxation_spec=asdict(request.spec),
                requested_occupancy=list(request.occupancy)))
        result['physical_batches'].append(record)
        if output is not None: save_json(output/filename,record)
        values = []
        try:
            # Raw cap is checked before the actual batch by the model prehook.
            # relax_actual may equal its cap here: these attempts are prepaid.
            with track_phase(ledger,'physical_batch',batch_id=batch_id,batch_size=len(requests),
                             kind=phase['name'],**{k:v for k,v in phase.items() if k!='name'}):
                values = list(physics.relax_many(requests,ledger.raw,max_steps=300))
            if len(values) != len(requests): raise RuntimeError('Physical batch returned incorrect lane count')
            # Persist every returned lane before validating any one lane. A bad
            # first lane must not hide the other three already computed E/Fs.
            from claude10_data import serialize_result
            for value,lane in zip(values,record['lanes']):
                raw_result = serialize_result(value)
                lane['result'] = dict(energy_eV=raw_result['energy_eV'],forces_eV_A=raw_result['forces'],
                    converged=raw_result['converged'],positions_A=raw_result['positions'],
                    numbers=raw_result['numbers'],cell_A=raw_result['cell'],pbc=raw_result['pbc'])
                lane['physics_evidence'] = dict(getattr(value,'evidence',{}))
            ledger.raw.check_time()
            lane_cost = 0
            for request,value,lane in zip(requests,values,record['lanes']):
                evidence = lane['physics_evidence']
                raw = evidence.get('raw_ef_attempts')
                if (evidence.get('formal_receipt') is not True or evidence.get('status') != 'converged'
                        or type(raw) is not int or raw < 1
                        or type(evidence.get('nsteps')) is not int or not 0 <= evidence['nsteps'] <= 300):
                    raise RuntimeError('Formal converged BFGS receipt and per-lane costs required')
                lane_cost += raw
                terminal = env.verify_result(value,request.spec)
                lane.update(status='verified_converged',terminal=_terminal_evidence(terminal))
            if counters.raw_ef-before['raw_ef'] != lane_cost or counters.forward_calls <= before['forward_calls']:
                raise RuntimeError('Physical batch raw E/F cost does not reconcile with lane receipts')
            record['status'] = 'verified_converged'
            return values
        except BaseException as exc:
            record.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
            evidence = getattr(exc,'evidence',getattr(physics,'last_evidence',[]))
            for lane,proof in zip(record['lanes'],evidence):
                lane.setdefault('physics_evidence',proof)
            raise
        finally:
            record.update(counts_after=asdict(counters),end_elapsed_seconds=clock()-ledger.started_at,
                          resources=old._resources())
            record['raw_ef_actual'] = counters.raw_ef-before['raw_ef']
            record['actual_forward_calls'] = counters.forward_calls-before['forward_calls']
            if output is not None: save_json(output/filename,record)

    try:
        # Legality gate has generation only; it never calls reference or MACE.
        for index,row in enumerate(heldout):
            ledger.check()
            occupancy = env.validate_occupancy(row['occupancy'])
            messages = row.get('messages')
            if messages is None: messages = env.encode_prompt_v2(unscored_state(occupancy,limits=limits,counters=counters))
            with track_phase(ledger,'generation_legality',index=index):
                action = actor.generate(messages,ledger,phase='legality',occupancy=occupancy)
            try:
                env.apply_action(occupancy,env.parse_action(action.text))
                legal = True
            except env.ActionError: legal = False
            row_record = dict(index=index,text=action.text,legal=legal,new_tokens=sum(action.action_mask))
            result['legality']['generations'].append(row_record)
            result['legality']['legal'] += int(legal)
            old._append(None if output is None else output/'generations.jsonl',dict(phase='legality',**row_record))
        if result['legality']['legal'] < 19:
            result['status'] = 'NOT_RELEASED_LEGALITY_GATE'
        else:
            ledger.check()
            requests = [env.build_request(occupancy) for occupancy in starts]
            if counters.relax_actual+4 > limits.max_relax_actual:
                raise env.BudgetExceeded('Four initial relaxations exceed cap')
            counters.relax_actual += 4
            result['initial_relaxations'] = 4
            initial_results = execute_batch(requests)
            initial = []
            for request,value in zip(requests,initial_results):
                terminal = cache.record(('review10',SEED),request,value)
                initial.append(terminal.result)
                initial_keys.add(terminal.mapping.canonical_key)
            history.update(initial_keys)
            for group_id,start in enumerate(initial):
                group = env.NoveltyGroup(history)
                environments = [env.Environment(start,None,arm='review10',seed=SEED,limits=limits,counters=counters,
                    cache=cache,frozen_history=group.frozen_history,started_at=ledger.started_at,clock=clock)
                    for _ in range(GROUP_SIZE)]
                details = [dict(group=group_id,rollout=i,generated=[],steps=[],samples=[]) for i in range(GROUP_SIZE)]
                result['partial_episodes'] = details
                for step in range(ATTEMPTS):
                    phase.clear(); phase.update(name='rollout',group=group_id,step=step)
                    actions = []
                    # All HF work is serialized on main before physical barrier.
                    for lane,environment in enumerate(environments):
                        ledger.check()
                        messages = env.encode_prompt_v2(environment.state)
                        with track_phase(ledger,'generation_rollout',group=group_id,rollout=lane,step=step):
                            action = actor.generate(messages,ledger,phase='rollout',occupancy=environment.state.occupancy)
                        generated = dict(step=step,text=action.text,action=asdict(action),status='generated')
                        details[lane]['generated'].append(generated)
                        old._append(None if output is None else output/'generations.jsonl',
                                    dict(phase='rollout',group=group_id,rollout=lane,**generated))
                        with track_phase(ledger,'reference_scoring',group=group_id,rollout=lane,step=step):
                            scored = actor.score_reference(action,ledger)
                        if (scored.text != action.text or scored.input_ids != action.input_ids
                                or scored.action_mask != action.action_mask or scored.old_logprobs != action.old_logprobs):
                            raise RuntimeError('Reference scorer changed immutable generated action/old policy record')
                        sample = scored.sample()
                        details[lane]['samples'].append(sample)
                        generated.update(status='reference_scored',sample=asdict(sample),record_sha256=sample.sha256)
                        actions.append(scored)
                    outcomes,failure = _StepBarrier(environments,ledger,execute_batch).run(actions)
                    for lane,(environment,outcome) in enumerate(zip(environments,outcomes)):
                        if outcome is None: continue
                        sample = details[lane]['samples'][-1]
                        record = dict(group=group_id,rollout=lane,step=step,text=actions[lane].text,
                            outcome=asdict(outcome),sample=asdict(sample),record_sha256=sample.sha256,
                            terminal_key_after=environment.state.terminal_key,counts=asdict(counters))
                        details[lane]['steps'].append(record)
                        old._append(None if output is None else output/'rollouts.jsonl',record)
                    if failure is not None: raise failure
                    if any(outcome is None or outcome.status=='physical_failure' for outcome in outcomes):
                        raise RuntimeError('Physical rollout failure stops incomplete group without update')
                group_returns, group_episodes = [], []
                for environment,detail in zip(environments,details):
                    steps = detail['steps']
                    detail.update(energy_return=sum(s['outcome']['energy_reward'] for s in steps),
                        novelty_return=sum(s['outcome']['novelty_reward'] for s in steps),
                        invalid_return=sum(s['outcome']['reward'] for s in steps if not s['outcome']['success']),
                        illegal_return=sum(s['outcome']['reward'] for s in steps if s['outcome']['status']=='illegal'),
                        physical_failure_return=0.,novelty_events=sum(s['outcome']['novelty_reward']>0 for s in steps))
                    detail['return'] = sum(s['outcome']['reward'] for s in steps)
                    group_returns.append(detail['return'])
                    group_episodes.append(detail['samples'])
                    group.register(environment.state)
                group.merge()
                episodes.extend(group_episodes); returns.append(group_returns)
                # Samples are serialized below; training retains immutable objects.
                result['groups'].append(dict(group=group_id,returns=group_returns,episodes=details,
                    advantages=group_advantages([group_returns])[0].tolist(),
                    frozen_history=sorted(group.frozen_history),merged_history=sorted(history)))
                result['partial_episodes'] = []
            advantages = group_advantages(returns)
            if not np.count_nonzero(advantages.numpy()):
                result.update(status='NO_UPDATE_ZERO_GROUP_VARIANCE',update=dict(optimizer_updates=0,
                    advantages=advantages.tolist(),gradient_status='not_measured_no_update'))
            else:
                ledger.check()
                before_hashes = [[sample.sha256 for sample in episode] for episode in episodes]
                with track_phase(ledger,'policy_update'):
                    update = actor.update(episodes,returns,ledger)
                if (update.get('optimizer_updates') != 1 or update.get('finite_gradients') is not True
                        or not math.isfinite(update.get('parameter_delta_l2',0)) or update.get('parameter_delta_l2',0) <= 0):
                    raise RuntimeError('One finite-gradient update with actual parameter change required')
                if before_hashes != [[s.sha256 for s in e] for e in episodes]:
                    raise RuntimeError('Immutable old/reference/mask records changed during joint update')
                update['gradient_status'] = 'measured_finite'
                result.update(status='COMPLETED_ONE_UPDATE',update=update,optimizer_updates=1)
            ledger.check()
            with track_phase(ledger,'adapter_save_unload_reload'):
                result['adapter_check'] = actor.finalize(ledger,output)
            if result['adapter_check'].get('save_unload_reload') is not True:
                raise RuntimeError('Policy adapter save/unload/disk reload failed')
            ledger.check()
            result['usable_for_workflow'] = True
    except env.BudgetExceeded as exc:
        result.update(status='BUDGET_EXHAUSTED',error=str(exc),usable_for_workflow=False)
    except Exception as exc:
        result.update(status='FAILED',error=type(exc).__name__+': '+str(exc),
                      traceback=traceback.format_exc(),usable_for_workflow=False)
    finally:
        for name,dependency in (('actor',actor),('physics',physics)):
            try:
                if hasattr(dependency,'close'):
                    with track_phase(ledger,name+'_close'): dependency.close()
            except Exception as exc:
                result.update(status='FAILED',usable_for_workflow=False)
                result[name+'_close_error'] = str(exc)
        # Remove object-only training views before JSON serialization.
        for detail in result['partial_episodes']+[e for g in result['groups'] for e in g['episodes']]:
            detail['samples'] = [asdict(s)|dict(sha256=s.sha256) for s in detail['samples']]
        result.update(counters=asdict(counters),cache_diagnostics=cache.diagnostics,
            N_unique=len(history-initial_keys),initial_terminal_keys=sorted(initial_keys),
            merged_terminal_keys=sorted(history),cost_events=ledger.events,phase_timings=ledger.phase_timings,
            resources=old._resources(),wall_seconds=clock()-ledger.started_at)
        if result['wall_seconds'] >= limits.wall_seconds:
            result.update(status='BUDGET_EXHAUSTED',usable_for_workflow=False,error='Stage cap during cleanup')
        if output is not None:
            _persist_result(output/'result.json',result,ledger)
    return result


def _persist_result(path,result,ledger):
    begin = ledger.clock()
    for _ in range(2):
        save_json(path,result)
        result['wall_seconds'] = ledger.clock()-ledger.started_at
        result['result_persistence'] = dict(seconds=ledger.clock()-begin,
            timing_boundary='post-write checked; final metadata write remains cooperative IO')
        if result['wall_seconds'] >= ledger.limits.wall_seconds:
            result.update(status='BUDGET_EXHAUSTED',usable_for_workflow=False,
                          error='Stage cap during final result persistence')
    if result['status'] == 'BUDGET_EXHAUSTED': save_json(path,result)


def _json(path):
    def invalid(value): raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(Path(path).read_text(),parse_constant=invalid)


def verify_release_manifest(path):
    """Validate ``review10-release-v1`` without loading a model or writing files.

    All file references are {path,sha256}; paths are relative to this manifest
    (``..`` is allowed for recovered evidence). Required top-level fields:

    * seed=2026102301, source_hashes=source_hashes(), authorization={review:10,
      retry_of:"650", authorized_retry_index:1, max_card_hours:2,
      additional_card_hours:6}; data_manifest, qwen_manifest, checkpoint refs.
    * sft, teacher652, mace648: {job_id,terminal:"COMPLETED",recovered:true,
      result:ref,verification:ref}. Every verification says independently_verified.
      SFT must be a NEW job; its proof has finite_gradients, parameters_changed,
      adapter_reload.{unloaded_without_merge,reloaded_from_disk,state_exact_match}
      true, legality.{denominator:80,attempts:80,legal:>=76}, and adapter_files
      [ref,...] including adapter_config.json and adapter_model.safetensors.
      Recovered SFT text is rechecked against all 80 frozen legality occupations;
      result training gradients, parameter fingerprints and exact reload agree.
    * teacher652 proof: job_id="652", formal_batch_bfgs_verified and
      all_240_receipts_verified true, complete_trajectories=16,
      attempts_per_trajectory=8, relax_actual=240, raw_ef and forward_calls equal
      result.counters, parent_exit_code="0:0". Existing semantic recovery is run
      on every formal receipt and trajectory; frozen teacher bytes must agree.
    * mace648 proof: job_id="648", raw_ef=1023, singleton_batch_parity.4/.16
      passed; original result raw_ef_calls=1023 and parity/checkpoint agree.
    * cpu_preflight:{result:ref,verification:ref}. Proof requires CPU_only,
      same_production_code_path true, batch_sizes=[1,4]. Result is produced by
      real_cpu_preflight: actual native/batch SP parity, BFGS one-step1/4,
      actual_forward_audit, old physics sources plus runner_source_hashes.
    * root_fees:{verification:ref}; proof requires independently_verified,
      accounting_complete, all_prior_jobs_terminal true,
      accounting_unknown_job_ids=[], review10_grpo_submissions=0,
      review10_actual_plus_remaining_caps_including_this<=6 and
      root_actual_plus_remaining_caps_including_this<=10 (finite card hours).

    Returns the existing HFActor release contract (verified, adapter_hashes),
    plus data/qwen/checkpoint/manifest hashes and evidence references. This
    checker grants no submission permission and creates no frozen job inputs.
    """
    from claude10_data import load_frozen_data, source_hashes as data_sources, verify_teacher_result
    from claude10_physics import source_hashes as physics_sources
    from smoke_sft10 import CONFIG
    path = Path(path).resolve()
    root = path.parent
    manifest = _json(path)

    def checked(reference):
        if (not isinstance(reference,dict) or not isinstance(reference.get('path'),str)
                or Path(reference['path']).is_absolute() or not reference['path']
                or not isinstance(reference.get('sha256'),str) or len(reference['sha256']) != 64):
            raise ValueError('Release file references require relative path and SHA256')
        target = root/reference['path']
        if sha256(target) != reference['sha256']:
            raise ValueError('Release evidence hash mismatch: '+reference['path'])
        return target

    def stage(name,expected_job=None):
        entry = manifest[name]
        job = str(entry.get('job_id',''))
        if (not job.isdigit() or (expected_job is not None and job != expected_job)
                or entry.get('terminal') != 'COMPLETED' or entry.get('recovered') is not True):
            raise ValueError('Recovered completed stage required: '+name)
        result = _json(checked(entry['result']))
        proof = _json(checked(entry['verification']))
        if result.get('status') != 'completed' or proof.get('independently_verified') is not True:
            raise ValueError('Independent successful verification required: '+name)
        return entry,result,proof

    def positive(value):
        return type(value) in (int,float) and math.isfinite(value) and value > 0

    try:
        if manifest.get('schema') != 'review10-release-v1' or manifest.get('seed') != SEED:
            raise ValueError('Review10 release schema/seed required')
        authorization = dict(review=10,retry_of='650',authorized_retry_index=1,max_card_hours=2,additional_card_hours=6)
        if any(manifest.get('authorization',{}).get(k) != v for k,v in authorization.items()):
            raise ValueError('Review10 unique retry authorization changed')
        if manifest.get('source_hashes') != source_hashes():
            raise ValueError('Review10 release source hashes changed')
        # Read/hash every declared release prerequisite before any model load.
        data_path = checked(manifest['data_manifest'])
        if data_path.name != 'manifest.json': raise ValueError('Frozen data manifest.json required')
        qwen_path = checked(manifest['qwen_manifest'])
        checkpoint_path = checked(manifest['checkpoint'])
        if manifest['checkpoint']['sha256'] != env.MODEL_SHA:
            raise ValueError('Frozen MACE checkpoint SHA changed')
        qwen = _json(qwen_path)
        if (qwen.get('repository') != 'Qwen/Qwen3-0.6B' or qwen.get('status') != 'verified_complete'
                or not any(f.get('path')=='model.safetensors' and f.get('sha256')==old.QWEN_WEIGHT_SHA
                           for f in qwen.get('files',[]))):
            raise ValueError('Verified official Qwen base manifest required')
        frozen = load_frozen_data(data_path.parent)
        sft_entry,sft,sft_proof = stage('sft')
        sft_job = str(sft_entry['job_id'])
        if sft_job in ('633','648','650','652') or str(sft.get('runtime',{}).get('slurm_job_id')) != sft_job:
            raise ValueError('New Review10 SFT job identity required')
        if (sft.get('schema') != 'review10-sft-v1' or sft.get('seed') != SEED
                or sft.get('config') != CONFIG or sft.get('source_hashes') != data_sources()
                or sft.get('GPU_work_performed') is not True or sft.get('physical_calls') != 0
                or sft.get('training_runs') != 1 or sft.get('validation_retries') != 0
                or sft.get('adapter_release_allowed') is not True
                or sft.get('data_manifest_sha256') != manifest['data_manifest']['sha256']
                or sft.get('qwen_manifest_sha256') != manifest['qwen_manifest']['sha256']):
            raise ValueError('New SFT configuration, successful gate and frozen data binding required')
        training = sft.get('training',{})
        updates = training.get('updates',[])
        if (sft_proof.get('finite_gradients') is not True or sft_proof.get('parameters_changed') is not True
                or training.get('parameters_changed') is not True or not updates
                or type(training.get('optimizer_updates')) is not int or training['optimizer_updates'] != len(updates)
                or not all(positive(row.get('gradient_norm')) for row in updates)
                or not training.get('adapter_hash_before') or not training.get('adapter_hash_after')
                or training['adapter_hash_before'] == training['adapter_hash_after']):
            raise ValueError('SFT finite positive gradients and real parameter changes required')
        reload_fields = ('unloaded_without_merge','reloaded_from_disk','state_exact_match')
        if not all(sft.get('reload',{}).get(k) is True and sft_proof.get('adapter_reload',{}).get(k) is True
                   for k in reload_fields):
            raise ValueError('SFT exact adapter disk reload required')
        adapter_hashes = {}
        adapter_paths = {}
        for reference in sft_proof.get('adapter_files',[]):
            target = checked(reference)
            if target.name in adapter_hashes: raise ValueError('Duplicate adapter evidence filename')
            adapter_hashes[target.name] = reference['sha256']
            adapter_paths[target.name] = target
        if not {'adapter_config.json','adapter_model.safetensors'} <= set(adapter_hashes):
            raise ValueError('Both hash-bound SFT adapter files required')
        if (sft['reload'].get('adapter_sha256') != adapter_hashes['adapter_model.safetensors']
                or sft['reload'].get('adapter_config_sha256') != adapter_hashes['adapter_config.json']):
            raise ValueError('SFT reloaded adapter file digests disagree with release')
        config = _json(adapter_paths['adapter_config.json'])
        if (config.get('r') != 16 or config.get('lora_alpha') != 32 or config.get('lora_dropout') != .05
                or set(config.get('target_modules',[])) != {'q_proj','v_proj'}):
            raise ValueError('New SFT adapter configuration changed')
        legality = sft.get('legality',{})
        records = legality.get('records',[])
        legal = 0
        if len(records) != 80 or len(frozen['rows']['legality_sft.jsonl']) != 80:
            raise ValueError('Exactly eighty frozen SFT legality records required')
        for index,(record,row) in enumerate(zip(records,frozen['rows']['legality_sft.jsonl'])):
            if record.get('index') != index or record.get('status') != 'completed':
                raise ValueError('Completed SFT legality timeline required')
            try:
                env.apply_action(row['occupancy'],env.parse_action(record['text']))
                valid = True
            except env.ActionError: valid = False
            if record.get('legal') is not valid: raise ValueError('SFT legality text disagrees with recorded legality')
            legal += int(valid)
        if (legal < 76 or legality.get('status') != 'passed'
                or any(legality.get(k)!=v or sft_proof.get('legality',{}).get(k)!=v
                       for k,v in dict(denominator=80,attempts=80,legal=legal).items())):
            raise ValueError('SFT legality requires at least 76/80 without retries')
        teacher_entry,teacher,teacher_proof = stage('teacher652','652')
        if frozen.get('manifest',{}).get('files',{}).get('teacher_result.json') != teacher_entry['result']['sha256']:
            raise ValueError('Frozen teacher bytes differ from independently recovered teacher652')
        if (teacher.get('source_hashes') != data_sources() or teacher.get('diagnostic') is not False
                or teacher.get('runtime',{}).get('GPU_work_performed') is not True
                or str(teacher.get('runtime',{}).get('slurm_job_id')) != '652'
                or str(teacher_proof.get('job_id')) != '652'
                or teacher_proof.get('parent_exit_code') != '0:0'
                or teacher_proof.get('formal_batch_bfgs_verified') is not True
                or teacher_proof.get('all_240_receipts_verified') is not True
                or teacher_proof.get('complete_trajectories') != 16
                or teacher_proof.get('attempts_per_trajectory') != 8):
            raise ValueError('Independent successful formal batch BFGS teacher652 required')
        if (teacher_proof.get('relax_actual') != 240
                or any(teacher_proof.get(k) != teacher.get('counters',{}).get(k)
                       for k in ('relax_actual','raw_ef','forward_calls'))):
            raise ValueError('Teacher receipt costs do not reconcile')
        verify_teacher_result(teacher)
        _,mace,mace_proof = stage('mace648','648')
        if (str(mace.get('runtime',{}).get('job_id')) != '648'
                or str(mace_proof.get('job_id')) != '648'
                or mace.get('raw_ef_calls') != 1023 or mace_proof.get('raw_ef') != 1023
                or mace.get('checkpoint_sha256') != env.MODEL_SHA
                or not all(mace_proof.get('singleton_batch_parity',{}).get(k,{}).get('passed') is True
                           and mace.get('batch_equivalence',{}).get('per_batch_size',{}).get(k,{}).get('passed') is True
                           for k in ('4','16'))):
            raise ValueError('Original successful MACE648 raw1023/batch4/16 parity required')
        cpu = _json(checked(manifest['cpu_preflight']['result']))
        cpu_proof = _json(checked(manifest['cpu_preflight']['verification']))
        if (not all(cpu_proof.get(k) is True for k in ('independently_verified','CPU_only','same_production_code_path'))
                or cpu_proof.get('batch_sizes') != [1,4]
                or cpu.get('status') != 'cpu_execution_preflight_passed'
                or cpu.get('production_code_path') is not True
                or cpu.get('runtime',{}).get('device') != 'cpu'
                or cpu.get('runtime',{}).get('GPU_work_performed') is not False
                or cpu.get('short_receipts_for_training') is not False
                or cpu.get('checkpoint_sha256') != env.MODEL_SHA):
            raise ValueError('Actual formal-path CPU SP/native/short BFGS preflight required')
        if cpu.get('source_hashes') != physics_sources() or cpu.get('runner_source_hashes') != source_hashes():
            raise ValueError('CPU preflight physics/runner source hashes changed')
        audit = cpu.get('actual_forward_audit',{})
        if (audit.get('all_precharged') is not True or audit.get('matches_counter') is not True
                or audit.get('observed_raw_ef') != 22 or audit.get('observed_forward_calls') != 13
                or cpu.get('counters',{}).get('raw_ef') != 22 or cpu['counters'].get('forward_calls') != 13):
            raise ValueError('Actual CPU forward/precharge accounting required')
        parity = cpu.get('parity',[])
        difference_fields = ('batch_single_energy_difference_eV','batch_single_force_difference_eV_A',
                             'batch_native_energy_difference_eV','batch_native_force_difference_eV_A')
        if (cpu.get('parity_tolerance') != 1e-6 or len(parity) != 4
                or any(row.get('lane') != i or any(type(row.get(k)) not in (int,float)
                    or not math.isfinite(row[k]) or not 0 <= row[k] <= 1e-6 for k in difference_fields)
                       for i,row in enumerate(parity))):
            raise ValueError('CPU native/singleton/batch4 SP parity failed')
        for size in (1,4):
            lanes = cpu.get('relaxations',{}).get(str(size),[])
            if len(lanes) != size or any(r.get('nsteps') != 1 or r.get('raw_ef_attempts') != 2
                    or r.get('execution_max_steps') != 1 or r.get('formal_receipt') is not False for r in lanes):
                raise ValueError('Actual CPU BFGS one-step batch1/4 proof required')
        fees = _json(checked(manifest['root_fees']['verification']))
        if (not all(fees.get(k) is True for k in ('independently_verified','accounting_complete','all_prior_jobs_terminal'))
                or fees.get('accounting_unknown_job_ids') != [] or fees.get('review10_grpo_submissions') != 0):
            raise ValueError('Closed root accounting and no unknown jobs/prior Review10 GRPO retry required')
        for key,bound in (('review10_actual_plus_remaining_caps_including_this',6),
                          ('root_actual_plus_remaining_caps_including_this',10)):
            value = fees.get(key)
            if type(value) not in (int,float) or not math.isfinite(value) or not 2 <= value <= bound:
                raise ValueError('Root accounting cannot reserve this unique two-card-hour retry')
        return dict(verified=True,manifest_sha256=sha256(path),adapter_hashes=adapter_hashes,
            source_hashes=source_hashes(),data_manifest_sha256=manifest['data_manifest']['sha256'],
            qwen_manifest_sha256=manifest['qwen_manifest']['sha256'],checkpoint_sha256=manifest['checkpoint']['sha256'],
            evidence={name:manifest[name] for name in ('sft','teacher652','mace648','cpu_preflight','root_fees')})
    except (KeyError,TypeError,OSError) as exc:
        raise ValueError('Incomplete or unreadable Review10 release evidence: '+str(exc)) from exc


def real_cpu_preflight(checkpoint,output):
    """Explicit CPU-only same-code-path SP/native parity and short BFGS1/4."""
    from claude10_physics import cpu_execution_preflight
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    before = source_hashes()
    proof = cpu_execution_preflight(checkpoint,output)
    if source_hashes() != before: raise RuntimeError('Runner sources changed during CPU preflight')
    proof['runner_source_hashes'] = before
    save_json(output/'result.json',proof)
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','adapter','qwen-manifest','checkpoint','release-manifest','data','output'):
        parser.add_argument('--'+name,type=Path,required=name in ('checkpoint','output'))
    parser.add_argument('--cpu-execution-preflight',action='store_true')
    args = parser.parse_args(argv)
    if args.cpu_execution_preflight:
        if any(getattr(args,k) is not None for k in ('model','adapter','qwen_manifest','release_manifest','data')):
            parser.error('Explicit CPU physics preflight accepts checkpoint/output only')
        proof = real_cpu_preflight(args.checkpoint,args.output)
        print(json.dumps(dict(status=proof['status'],counters=proof.get('counters'),CPU_only=True)))
        return
    if any(getattr(args,k) is None for k in ('model','adapter','qwen_manifest','release_manifest','data')):
        parser.error('In-job runner requires model/adapter/qwen-manifest/release-manifest/data')
    if args.output.exists(): raise FileExistsError(args.output)
    started_at = PROCESS_STARTED_AT
    release = verify_release_manifest(args.release_manifest)
    for target,key in ((args.data/'manifest.json','data_manifest_sha256'),
                       (args.qwen_manifest,'qwen_manifest_sha256'),(args.checkpoint,'checkpoint_sha256')):
        if sha256(target) != release[key]: raise ValueError('Requested runtime input differs from released '+key)
    old.verify_adapter_binding(args.adapter,release)
    from claude10_data import load_frozen_data
    frozen = load_frozen_data(args.data)
    runtime = old.require_single_gpu_job()
    if 'A100' not in runtime.get('device_name',''): raise RuntimeError('Review10 requires one Slurm CUDA A100')
    actor = HFActor(args.model,args.adapter,args.qwen_manifest,release=release,output=args.output)
    # Backend creates only its own subdirectory. run_smoke owns the new root.
    class LazyPhysics:
        def __init__(self): self.engine = None
        def relax_many(self,requests,counter,max_steps=300):
            if self.engine is None:
                self.engine = BatchMACEPhysics(args.checkpoint,args.output/'physics_engine',device='cuda',actor=actor)
            return self.engine.relax_many(requests,counter,max_steps=max_steps)
        @property
        def last_evidence(self): return [] if self.engine is None else self.engine.last_evidence
        def close(self):
            if self.engine is not None: self.engine.close()
    physics = LazyPhysics()
    result = run_smoke(actor,physics,[r['occupancy'] for r in frozen['rows']['starts.jsonl']],
        frozen['rows']['legality_grpo.jsonl'],output=args.output,started_at=started_at,
        runtime_context=dict(runtime=runtime,release=release,data_manifest_sha256=frozen['manifest_sha256']))
    print(json.dumps({k:result[k] for k in ('status','usable_for_workflow','counters','wall_seconds')}))
    if not result['usable_for_workflow']: raise SystemExit(1)


if __name__ == '__main__': main()
