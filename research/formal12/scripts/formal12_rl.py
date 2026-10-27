"""Review12 formal jobs: d0sft, LLM arm rounds, SA search rounds, blind rollouts.

One CLI, one job per (seed, stage). All stages share environment v2.2, batch
MACE physics and the Review11 constrained actor. Import starts no work.
"""
from __future__ import annotations

import time
PROCESS_STARTED_AT = time.monotonic()
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

import claude08_env as env
from claude08_grpo import group_advantages
from formal12_common import (SCHEMA, SEEDS, ARMS, ROUNDS, GROUPS_PER_ROUND, GROUP_SIZE, K, UPDATE_EVERY_GROUPS,
                             D0_RELAX_CAP, ROUND_RELAX_CAP, BLIND_RELAX_CAP, LIMITS, Ledger, Phase, append_jsonl,
                             save_json, sha256, source_hashes, start_pools, stream, runtime_info, make_physics)
import formal12_search as search
from smoke_grpo10 import _StepBarrier

DIAGNOSTIC_PER_ROUND = 5
ENCODE = env.encode_prompt_v2          # Review12.3: --prompt v3 switches to formal12_prompt.encode_prompt_v3


# ----------------------------------------------------------------------------- shared helpers

def terminals_of(cache, namespace):
    """Every distinct verified terminal in this job's namespace (lowest-energy geometry per key)."""
    best = {}
    for (protocol_sha, ns, key), records in cache._terminals.items():
        if ns != tuple(namespace): continue
        record = min(records, key=lambda r: r.result.energy)
        best[key] = record
    return best


def terminal_rows(terminals):
    rows = []
    for key, record in sorted(terminals.items(), key=lambda kv: (kv[1].result.energy, kv[0])):
        atoms = record.result.atoms
        rows.append(dict(terminal_key=key, energy_eV=float(record.result.energy),
                         geometry_hash=record.mapping.geometry_hash, occupancy=list(record.mapping.occupancy),
                         shell_counts=record.mapping.shell_counts, near=bool(record.mapping.quality_metadata.get('NEAR_THRESHOLD', False)),
                         numbers=atoms.numbers.tolist(), positions_A=atoms.positions.tolist(), cell_A=atoms.cell.array.tolist()))
    return rows


def pick_diagnostics(rows, seed, label, k=DIAGNOSTIC_PER_ROUND):
    """Preregistered: 3 lowest, 1 hash-random, 1 shell-farthest from the lowest (distinct keys)."""
    if not rows: return []
    ordered = sorted(rows, key=lambda r: (r['energy_eV'], r['terminal_key']))
    chosen = ordered[:3]
    rest = [r for r in ordered[3:]]
    if rest:
        pick = min(rest, key=lambda r: hashlib.sha256((label + r['terminal_key']).encode()).hexdigest())
        chosen.append(pick); rest.remove(pick)
    if rest:
        vector = lambda r: np.array([r['shell_counts']['Y_vac_NN'], r['shell_counts']['Y_vac_NNN']], dtype=float)
        far = max(rest, key=lambda r: (float(np.abs(vector(r) - vector(chosen[0])).sum()), r['terminal_key']))
        chosen.append(far)
    return [r['terminal_key'] for r in chosen[:k]]


def load_history(paths):
    keys = set()
    for path in paths:
        keys.update(json.loads(Path(path).read_text())['merged_terminal_keys'])
    return keys


class Physics:
    """execute_batch for _StepBarrier on the main thread."""

    def __init__(self, engine, ledger, log):
        self.engine, self.ledger, self.log = engine, ledger, log

    def __call__(self, requests):
        from formal12_common import verify_physics_lanes
        before = self.ledger.counters.raw_ef
        with Phase(self.ledger, 'physical_batch', size=len(requests)):
            values = list(self.engine.relax_many(requests, self.ledger.raw, max_steps=300))
        if len(values) != len(requests): raise RuntimeError('Physics returned wrong lane count')
        verify_physics_lanes(requests, values, before, self.ledger)
        for request, value in zip(requests, values):
            append_jsonl(self.log, dict(request_key=request.key, energy_eV=float(value.energy),
                                        nsteps=getattr(value, 'evidence', {}).get('nsteps')))
        return values


class FormalEnvironment(env.Environment):
    """v2.2 Environment; Review12.1 turns a step-capped unconverged relaxation into a paid, rolled-back,
    zero-reward UNCONVERGED step (as UNMAPPABLE) instead of a job-stopping physical failure."""

    def __init__(self, *args, ledger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ledger = ledger

    def step(self, text, **kwargs):
        try:
            return super().step(text, **kwargs)
        except env.PhysicalFailure as exc:
            cause = exc.__cause__
            if not (isinstance(cause, ValueError) and 'did not converge' in str(cause)): raise
            self._physical_stop = None
            if self.state.history and self.state.history[-1].status == 'physical_failure': self.state.history.pop()
            request_key = exc.step_result.request_key if exc.step_result is not None else None
            self.state.history.append(env.HistoryEntry(text, 'UNCONVERGED', None, request_key, None, 0.))
            if self.ledger is not None: self.ledger.note_unconverged()
            return env.StepResult(False, 'UNCONVERGED', 0., 0., 0., error=str(cause), request_key=request_key)


def make_env(terminal, ledger, cache, namespace, frozen_history, seed):
    return FormalEnvironment(terminal.result, None, arm=namespace[0], seed=seed, limits=ledger.limits,
                             counters=ledger.counters, cache=cache, frozen_history=frozen_history,
                             started_at=ledger.started_at, clock=ledger.clock, ledger=ledger)


# ----------------------------------------------------------------------------- LLM rollouts

MENU = False                           # Review12.3 step 6: --prompt v4 (candidate menu); set in main()


class Carrier:
    """Minimal object for _StepBarrier: the environment only needs the action text."""
    def __init__(self, text): self.text = text


def choose(actor, environment, ledger, *, seed, lane, step, group_id, score_reference):
    """Returns (env_text, action_for_training_or_None, messages, record)."""
    state = environment.state
    if not MENU:
        messages = ENCODE(state)
        with Phase(ledger, 'generation', group=group_id, step=step):
            action = actor.generate(messages, ledger, phase='rollout', occupancy=state.occupancy)
        if score_reference:
            with Phase(ledger, 'reference_scoring'):
                action = actor.score_reference(action, ledger)
        return action.text, action, messages, {}
    from formal12_menu import make_menu, encode_prompt_v4, greedy_choice
    menu = make_menu(state.occupancy, (seed, state.terminal_key, state.steps_taken))
    messages = encode_prompt_v4(state, menu)
    if isinstance(actor, str):   # non-LLM menu baselines on the identical menu
        rng = np.random.default_rng([seed, group_id, lane, step])
        picked = greedy_choice(menu, rng) if actor == 'greedy' else menu[int(rng.integers(len(menu)))]
        return picked['action'].to_json(), None, messages, dict(choice=picked['k'], dNN=picked['dNN'], baseline=actor)
    actor.menu = menu
    try:
        with Phase(ledger, 'generation', group=group_id, step=step):
            action = actor.generate(messages, ledger, phase='rollout', occupancy=state.occupancy)
    finally:
        actor.menu = None
    if score_reference:
        with Phase(ledger, 'reference_scoring'):
            action = actor.score_reference(action, ledger)
    picked = menu[json.loads(action.text)['choice']]
    return picked['action'].to_json(), action, messages, dict(choice=picked['k'], dNN=picked['dNN'],
                                                              min_dNN=min(m['dNN'] for m in menu))


def rollout_group(actor, environments, ledger, execute, *, score_reference, group_id, journal, seed=0):
    """G4 synchronized K-step trajectories; returns per-lane samples, steps, return."""
    lanes = [dict(samples=[], steps=[], texts=[]) for _ in environments]
    for step in range(K):
        carriers, actions = [], []
        for lane, environment in enumerate(environments):
            ledger.check()
            text, action, messages, extra = choose(actor, environment, ledger, seed=seed, lane=lane, step=step,
                                                   group_id=group_id, score_reference=score_reference)
            lanes[lane]['messages_' + str(step)] = messages
            lanes[lane].setdefault('extras', []).append(extra)
            carriers.append(Carrier(text)); actions.append(action)
        outcomes, failure = _StepBarrier(environments, ledger, execute).run(carriers)
        if failure is not None: raise failure
        for lane, (environment, outcome, action, carrier) in enumerate(zip(environments, outcomes, actions, carriers)):
            if outcome is None or outcome.status == 'physical_failure':
                raise RuntimeError('Physical rollout failure stops the job')
            row = dict(group=group_id, lane=lane, step=step, text=carrier.text, outcome=asdict(outcome),
                       energy_eV=environment.state.energy, best_energy_eV=environment.state.best_energy,
                       **lanes[lane]['extras'][-1])
            lanes[lane]['steps'].append(row)
            if action is not None: lanes[lane]['samples'].append(action.sample())
            append_jsonl(journal, row)
    for lane in lanes:
        lane['return'] = sum(s['outcome']['reward'] for s in lane['steps'])
    return lanes


def run_llm_round(arm, actor, engine, starts, ledger, *, seed, round_index, history, output, prior_rows=None,
                  blind=False, spares=(), update_every=UPDATE_EVERY_GROUPS, itersft_epochs=2):
    """One round (16 starts) of an LLM arm; trajectory or step-level GRPO, iterated SFT, or frozen."""
    namespace = (arm, seed)
    cache = env.RequestCache()
    journal = Path(output) / 'rollouts.jsonl'
    execute = Physics(engine, ledger, Path(output) / 'physics.jsonl')
    with Phase(ledger, 'initial_relaxations'):
        initials = search.relax_initials(engine, cache, namespace, starts, ledger, log=Path(output) / 'initial.jsonl',
                                         spares=spares)
    train = not blind and arm in ('grpo_traj', 'grpo_step')
    groups, pending_episodes, pending_returns, updates = [], [], [], []
    rng = stream(seed, f'{arm}-round{round_index}-branch')
    for group_id, start in enumerate(initials):
        novelty = env.NoveltyGroup(history)
        if arm == 'grpo_step' and not blind:
            record, episodes, returns = step_group(actor, start, ledger, cache, namespace, novelty, execute, rng,
                                                   seed=seed, group_id=group_id, journal=journal)
            pending_episodes.extend(episodes); pending_returns.extend(returns)
        else:
            environments = [make_env(start, ledger, cache, namespace, novelty.frozen_history, seed) for _ in range(GROUP_SIZE)]
            lanes = rollout_group(actor, environments, ledger, execute, score_reference=train, group_id=group_id,
                                  journal=journal, seed=seed)
            for environment in environments: novelty.register(environment.state)
            novelty.merge()
            returns = [lane['return'] for lane in lanes]
            record = dict(group=group_id, returns=returns, start_energy_eV=start.result.energy,
                          best_energy_eV=[min([start.result.energy] + [s['best_energy_eV'] for s in lane['steps']]) for lane in lanes],
                          lanes=[dict(steps=lane['steps']) for lane in lanes])
            if train:
                pending_episodes.extend(lane['samples'] for lane in lanes); pending_returns.append(returns)
            if arm == 'itersft' and not blind:
                best = max(range(GROUP_SIZE), key=lambda i: (returns[i], -int(hashlib.sha256(
                    json.dumps(lanes[i]['steps'], sort_keys=True).encode()).hexdigest(), 16)))
                record['selected_lane'] = best
                record['selected_rows'] = [dict(messages=lanes[best]['messages_' + str(s['step'])], group=group_id, step=s['step'],
                                                **({'choice': s['choice']} if MENU else {'action': json.loads(s['text'])}))
                                           for s in lanes[best]['steps'] if s['outcome']['status'] == 'success']
        groups.append(record)
        if train and (group_id + 1) % update_every == 0:
            returns_arr = np.array(pending_returns, dtype=float)
            if returns_arr.ndim == 1: returns_arr = returns_arr.reshape(-1, GROUP_SIZE)
            with Phase(ledger, 'policy_update', after_group=group_id):
                updates.append(actor.update(pending_episodes, returns_arr.tolist(), ledger))
            pending_episodes, pending_returns = [], []
    sft = None
    if arm == 'itersft' and not blind:
        new_rows = [row for g in groups for row in g['selected_rows']]
        rows = mix_replay(new_rows, prior_rows or [], seed=seed, round_index=round_index)
        with Phase(ledger, 'itersft_train', rows=len(rows)):
            sft = actor.sft_policy(rows, ledger, epochs=itersft_epochs, seed=seed + round_index,
                                   log_path=Path(output) / 'itersft_updates.json')
        save_json(Path(output) / 'selected_rows.json', new_rows)
    terminals = terminals_of(cache, namespace)
    history.update(terminals)
    return dict(groups=groups, updates=updates, itersft=sft, terminals=terminal_rows(terminals),
                cache_diagnostics=cache.diagnostics)


def step_group(actor, start, ledger, cache, namespace, novelty, execute, rng, *, seed, group_id, journal):
    """Step-level GRPO: at each of K states, 4 one-step branches; continue on a preregistered random branch."""
    current, best = start, start.result.energy
    visits, history_entries = set(novelty.frozen_history) | {start.mapping.canonical_key}, []
    episodes, returns, steps = [], [], []
    for step in range(K):
        frozen = frozenset(visits)
        environments = [make_env(current, ledger, cache, namespace, frozen, seed) for _ in range(GROUP_SIZE)]
        for environment in environments:
            environment.state.best_energy = best
            environment.state.history = list(history_entries)
            environment.state.steps_taken = step
        actions = []
        for environment in environments:
            ledger.check()
            messages = ENCODE(environment.state)
            with Phase(ledger, 'generation', group=group_id, step=step):
                action = actor.generate(messages, ledger, phase='rollout', occupancy=environment.state.occupancy)
            with Phase(ledger, 'reference_scoring'):
                actions.append(actor.score_reference(action, ledger))
        outcomes, failure = _StepBarrier(environments, ledger, execute).run(actions)
        if failure is not None: raise failure
        rewards = []
        for environment, outcome in zip(environments, outcomes):
            if outcome is None or outcome.status == 'physical_failure': raise RuntimeError('Physical failure')
            immediate = (current.result.energy - environment.state.energy) / .20 if outcome.success else 0.
            rewards.append(immediate + outcome.novelty_reward + (outcome.reward if outcome.status == 'illegal' else 0.))
        choice = int(rng.integers(GROUP_SIZE))
        chosen = environments[choice]
        row = dict(group=group_id, step=step, texts=[a.text for a in actions], rewards=rewards,
                   statuses=[o.status for o in outcomes], energies=[e.state.energy for e in environments], choice=choice)
        steps.append(row); append_jsonl(journal, row)
        episodes.extend([a.sample()] for a in actions); returns.append(rewards)
        visits.update(e.state.terminal_key for e in environments)
        history_entries = list(chosen.state.history)
        if outcomes[choice].success:
            records = cache.terminal_records(namespace, chosen.state.terminal_key)
            current = next(r for r in records if r.mapping.geometry_hash == chosen.state.geometry_hash)
            best = min(best, chosen.state.energy)
    novelty.history.update(visits)
    return dict(group=group_id, steps=steps, start_energy_eV=start.result.energy, best_energy_eV=best), episodes, returns


def mix_replay(new_rows, prior_rows, *, seed, round_index):
    """Iterated SFT data: all new rows plus an equal number of replayed prior rows (D0 and earlier rounds)."""
    rng = np.random.default_rng(seed * 10 + round_index)
    if not prior_rows: return list(new_rows)
    take = min(len(prior_rows), max(len(new_rows), 1))
    picked = [prior_rows[int(i)] for i in rng.choice(len(prior_rows), take, replace=False)]
    return list(new_rows) + picked


# ----------------------------------------------------------------------------- jobs

def job_d0sft(args, ledger, out):
    from formal12_actor import sft_from_base
    pools = start_pools(args.seed)
    engine = make_physics(args.checkpoint, out, args.device)
    cache, namespace = env.RequestCache(), ('d0', args.seed)
    with Phase(ledger, 'd0_collection'):
        chains, summaries, initials = search.collect_d0(engine, cache, namespace, pools['d0'], ledger, seed=args.seed,
                                                        log=out / 'physics.jsonl', spares=pools.get('spare', ()))
    engine.close() if hasattr(engine, 'close') else None
    rows = [row for chain in chains for row in chain.rows]
    save_json(out / 'd0_rows.json', rows)
    terminals = terminals_of(cache, namespace)
    with Phase(ledger, 'd0_sft'):
        sft = sft_from_base(args.model, args.qwen_manifest, rows, out, ledger, device=args.device,
                            epochs=args.sft_epochs, seed=args.seed, dtype=args.dtype)
    return dict(chains=summaries, d0_rows=len(rows), sft=sft, terminals=terminal_rows(terminals),
                merged_terminal_keys=sorted(terminals), diagnostics=pick_diagnostics(terminal_rows(terminals), args.seed, 'd0'))


def job_llm(args, ledger, out):
    from formal12_actor import FormalActor
    pools = start_pools(args.seed)
    blind = args.stage == 'blind'
    starts = (pool_union(pools, args.pool) if args.pool else pools['blind']) if blind else pool_union(pools, args.pool or f'round{args.round}')
    if args.test_groups: starts = starts[:args.test_groups]
    if args.arm in ('greedy', 'menurandom'):
        if not MENU: raise ValueError('Menu baselines require --prompt v4')
        engine = make_physics(args.checkpoint, out, args.device)
        result = run_llm_round(args.arm, args.arm, engine, starts, ledger, seed=args.seed, round_index=args.round,
                               history=load_history(args.history), output=out, blind=True, spares=pools.get('spare', ()))
        result['merged_terminal_keys'] = sorted(result['terminals'][i]['terminal_key'] for i in range(len(result['terminals'])))
        return result
    actor = FormalActor(args.model, args.reference_adapter, args.policy_adapter, args.qwen_manifest,
                        output=out, device=args.device, dtype=args.dtype)
    actor.grpo_lr, actor.sft_lr = args.grpo_lr, args.sft_lr
    actor.persistent_optimizer = args.persistent_optimizer
    actor.seed_for_generator = int(stream(args.seed, f'{args.arm}-{args.stage}-{args.round}-decode').integers(0, 2**63 - 1))
    engine = make_physics(args.checkpoint, out, args.device)
    history = load_history(args.history)
    prior_rows = []
    for path in args.prior_rows: prior_rows.extend(json.loads(Path(path).read_text()))
    if args.prompt == 'v3':
        from formal12_prompt import to_v3
        prior_rows = [dict(r, messages=to_v3(r['messages'])) if 'local' not in r['messages'][1]['content'] else r
                      for r in prior_rows]
    result = run_llm_round(args.arm, actor, engine, starts, ledger, seed=args.seed, round_index=args.round,
                           history=history, output=out, prior_rows=prior_rows, blind=blind, spares=pools.get('spare', ()),
                           update_every=args.update_every, itersft_epochs=args.itersft_epochs)
    from smoke_grpo11 import summarize
    result['constrained_diagnostics'] = summarize(actor.records)
    result['load_corrections'] = actor.load_corrections
    result['identical_adapters_at_load'] = actor.identical_adapters_at_load
    if not blind and args.arm != 'frozen':
        with Phase(ledger, 'adapter_save_unload_reload'):
            result['adapter_check'] = actor.finalize(ledger, out)
    result['merged_terminal_keys'] = sorted(history)
    result['diagnostics'] = pick_diagnostics(result['terminals'], args.seed, f'{args.arm}-{args.stage}-{args.round}')
    actor.close()
    return result


def job_sft(args, ledger, out):
    """SFT-only stage (Review12.3 step 5): fresh LoRA on --prior-rows (converted to the selected prompt)."""
    from formal12_actor import sft_from_base
    rows = []
    for path in args.prior_rows: rows.extend(json.loads(Path(path).read_text()))
    if args.prompt == 'v3':
        from formal12_prompt import to_v3
        rows = [dict(r, messages=to_v3(r['messages'])) for r in rows]
    elif args.prompt == 'v4':
        from formal12_menu import sft_rows_v4
        rows = sft_rows_v4(rows, seed=args.seed, copies=4)
    save_json(out / 'sft_rows.json', rows)
    sft = sft_from_base(args.model, args.qwen_manifest, rows, out, ledger, device=args.device, epochs=args.sft_epochs,
                        seed=args.seed, dtype=args.dtype)
    return dict(sft=sft, rows=len(rows), merged_terminal_keys=[])


def job_search(args, ledger, out):
    pools = start_pools(args.seed)
    starts = pools['blind'] if args.stage == 'blind' else (pool_union(pools, args.pool) if args.pool else
              pools['d0'] if args.round == 0 else pools[f'round{args.round}'])
    if args.test_groups: starts = starts[:args.test_groups]
    engine = make_physics(args.checkpoint, out, args.device)
    cache, namespace = env.RequestCache(), ('search', args.seed)
    history = load_history(args.history)
    with Phase(ledger, 'search'):
        chains, _ = search.run_search_round(engine, cache, namespace, starts, ledger, seed=args.seed,
                                            label=f'{args.stage}-{args.round}', log=out / 'physics.jsonl',
                                            chains_per_start=1 if (args.stage == 'search' and args.round == 0) else GROUP_SIZE,
                                            spares=pools.get('spare', ()),
                                            temperatures=tuple(map(float, np.linspace(args.sa_tmax, args.sa_tmin, K))))
    terminals = terminals_of(cache, namespace); history.update(terminals)
    rows = terminal_rows(terminals)
    return dict(chains=chains, terminals=rows, merged_terminal_keys=sorted(history),
                diagnostics=pick_diagnostics(rows, args.seed, f'search-{args.stage}-{args.round}'))


def pool_union(pools, spec):
    """'val+tune_8' -> concatenated start lists (Review12.3 larger validation set)."""
    out = []
    for name in spec.split('+'): out.extend(pools[name])
    return out


def relax_cap(args):
    # + SPARE_STARTS: a replaced initial state (unmappable/unconverged start) is paid but must not
    # cut the round short; the nominal per-round budget is unchanged otherwise (Review12.2).
    from formal12_common import SPARE_STARTS
    if args.stage == 'sft': return 0
    if args.stage == 'd0sft': return D0_RELAX_CAP + SPARE_STARTS
    if args.stage == 'blind':
        if args.pool: return len(pool_union(start_pools(args.seed), args.pool)) * (1 + GROUP_SIZE * K) + SPARE_STARTS
        return BLIND_RELAX_CAP + SPARE_STARTS
    if args.stage == 'search' and args.round == 0: return 52 * (1 + K) + SPARE_STARTS
    if args.stage == 'search' and args.pool:
        return len(pool_union(start_pools(args.seed), args.pool)) * (1 + GROUP_SIZE * K) + SPARE_STARTS
    return ROUND_RELAX_CAP + SPARE_STARTS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('d0sft', 'round', 'search', 'blind', 'sft'), required=True)
    parser.add_argument('--arm', choices=ARMS + ('search', 'greedy', 'menurandom'))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--qwen-manifest', type=Path)
    parser.add_argument('--reference-adapter', type=Path)
    parser.add_argument('--policy-adapter', type=Path)
    parser.add_argument('--history', type=Path, nargs='*', default=[])
    parser.add_argument('--prior-rows', type=Path, nargs='*', default=[])
    parser.add_argument('--sft-epochs', type=int, default=3)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--dtype', default=None)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test-groups', type=int, default=None, help='CPU tests only: use the first N starts')
    # Review12.3 tunable configuration (defaults = the original preregistered values)
    parser.add_argument('--pool', default=None, help='start pool override (tune_a, tune_b, round4, round5)')
    parser.add_argument('--prompt', choices=('v2', 'v3', 'v4'), default='v2')
    parser.add_argument('--grpo-lr', type=float, default=1e-5)
    parser.add_argument('--persistent-optimizer', action='store_true')
    parser.add_argument('--update-every', type=int, default=UPDATE_EVERY_GROUPS)
    parser.add_argument('--sft-lr', type=float, default=1e-4)
    parser.add_argument('--itersft-epochs', type=int, default=2)
    parser.add_argument('--sa-tmax', type=float, default=.20)
    parser.add_argument('--sa-tmin', type=float, default=.02)
    args = parser.parse_args(argv)
    if args.test_groups is not None and args.device != 'cpu': raise ValueError('--test-groups is CPU-test only')
    if args.seed not in SEEDS: raise ValueError('Unregistered seed')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    limits = replace(LIMITS, max_relax_actual=relax_cap(args))
    ledger = Ledger(limits, started_at=PROCESS_STARTED_AT if args.device == 'cuda' else None,
                    log_path=args.output / 'cost_events.jsonl')
    ledger.raw.phase_ledger = ledger
    result = dict(schema=SCHEMA, stage=args.stage, arm=args.arm, seed=args.seed, round=args.round,
                  protocol=env.PROTOCOL, protocol_sha=env.protocol_fingerprint(env.RelaxationSpec()),
                  source_hashes=source_hashes(), limits=asdict(limits), status='STARTED',
                  config=dict(pool=args.pool, grpo_lr=args.grpo_lr, update_every=args.update_every, sft_lr=args.sft_lr,
                              persistent_optimizer=args.persistent_optimizer, prompt=args.prompt,
                              itersft_epochs=args.itersft_epochs, sa_tmax=args.sa_tmax, sa_tmin=args.sa_tmin),
                  inputs={k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v)
                          for k, v in vars(args).items()})
    try:
        result['runtime'] = runtime_info(args.device)
        global ENCODE, MENU
        if args.prompt == 'v3':
            from formal12_prompt import encode_prompt_v3
            ENCODE = encode_prompt_v3
        if args.prompt == 'v4':
            MENU = True
        if args.stage == 'd0sft': result.update(job_d0sft(args, ledger, args.output))
        elif args.stage == 'sft': result.update(job_sft(args, ledger, args.output))
        elif args.stage == 'search' or args.arm == 'search': result.update(job_search(args, ledger, args.output))
        else: result.update(job_llm(args, ledger, args.output))
        result['status'] = 'COMPLETED'
    except env.BudgetExceeded as exc:
        result.update(status='BUDGET_EXHAUSTED', error=str(exc), traceback=traceback.format_exc())
    except Exception as exc:
        result.update(status='FAILED', error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
    finally:
        result.update(counters=asdict(ledger.counters), wall_seconds=ledger.elapsed(), phase_timings=ledger.phase_timings,
                      unconverged_relaxations=ledger.unconverged, replaced_starts=getattr(ledger, 'replaced_starts', []),
                      protocol_amendment='Review12.1 unconverged handling + spare starts')
        save_json(args.output / 'result.json', result)
    print(json.dumps({k: result.get(k) for k in ('status', 'stage', 'arm', 'seed', 'round', 'counters', 'wall_seconds', 'error')}))
    if result['status'] != 'COMPLETED': raise SystemExit(1)


if __name__ == '__main__': main()
