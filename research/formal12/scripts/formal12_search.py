"""Review12 search chains: D0 teacher (SA + random walk) and the SA baseline.

A chain holds one verified terminal state. Each step proposes one uniformly
random legal action, relaxes the requested occupation (batched with the other
chains of the same step), and accepts by Metropolis (SA) or always (random).
UNMAPPABLE terminals are paid, rejected and counted (v2.2 rule). SFT rows are
the downhill accepted moves, with the exact prompt the LLM would have seen.
"""
from __future__ import annotations
from dataclasses import asdict
import json

import numpy as np

import claude08_env as env
from formal12_common import (K, D0_MIX, Ledger, Phase, append_jsonl, stream, verify_physics_lanes)

TEMPERATURES = tuple(map(float, np.linspace(.20, .02, K)))


class Chain:
    def __init__(self, mode, terminal, *, chain_id, start_id, rng, namespace, ledger, temperatures=TEMPERATURES):
        if mode not in ('SA', 'random'): raise ValueError('Unknown search mode')
        self.mode, self.current, self.chain_id, self.start_id = mode, terminal, chain_id, start_id
        self.rng, self.namespace, self.ledger = rng, namespace, ledger
        self.best = terminal.result.energy
        self.visits = {terminal.mapping.canonical_key}
        self.history, self.steps, self.rows = [], [], []
        self.temperatures = tuple(temperatures)

    def state(self):
        """EpisodeState as the LLM sees it (same encode_prompt_v2 path)."""
        mapping, result = self.current.mapping, self.current.result
        return env.EpisodeState(mapping.occupancy, result.atoms.copy(), result.energy, self.best,
                                mapping.canonical_key, mapping.geometry_hash, frozenset(), set(self.visits),
                                self.ledger.limits, self.ledger.counters, self.namespace, self.ledger.started_at,
                                self.ledger.clock, steps_taken=len(self.steps), history=list(self.history))

    def propose(self):
        legal = env.enumerate_legal_actions(self.current.mapping.occupancy)
        action = legal[int(self.rng.integers(len(legal)))]
        uniform = float(self.rng.random())
        return action, uniform


def relax_batch(physics, cache, namespace, requests, ledger, *, log=None, meta=None):
    """Relax 1..4 requests (cache misses only); return VerifiedTerminal or exception per lane."""
    out, misses = [None] * len(requests), []
    for i, request in enumerate(requests):
        hit = cache.get(namespace, request)
        if hit is not None:
            ledger.counters.cache_hits += 1; out[i] = hit
        else:
            misses.append(i)
    if misses:
        ledger.charge_relax(len(misses))
        before = ledger.counters.raw_ef
        batch = [requests[i] for i in misses]
        with Phase(ledger, 'physical_batch', size=len(batch)):
            values = list(physics.relax_many(batch, ledger.raw, max_steps=300))
        if len(values) != len(batch): raise RuntimeError('Physics returned wrong lane count')
        verify_physics_lanes(batch, values, before, ledger)
        for i, value in zip(misses, values):
            try:
                out[i] = cache.record(namespace, requests[i], value)
            except env.UnmappableTerminal as exc:
                out[i] = exc
            except ValueError as exc:
                if value.converged is not False or 'did not converge' not in str(exc): raise
                out[i] = Unconverged(str(exc)); ledger.note_unconverged()
            append_jsonl(log, dict(meta[i] if meta else {}, request_key=requests[i].key,
                energy_eV=float(value.energy), status=('UNCONVERGED' if isinstance(out[i], Unconverged) else 'UNMAPPABLE')
                if isinstance(out[i], Exception) else 'mapped',
                nsteps=getattr(value, 'evidence', {}).get('nsteps'),
                terminal_key=None if isinstance(out[i], Exception) else out[i].mapping.canonical_key,
                mapping=env.mapping_diagnostics(value.atoms) if not isinstance(out[i], Exception) else None))
    return out


class Unconverged(Exception):
    pass


def relax_initials(physics, cache, namespace, occupancies, ledger, *, log=None, spares=()):
    """Relax starts in batches of four; an unmappable/unconverged start is replaced by the next spare (logged)."""
    terminals, spares, replaced = [], list(spares), []
    pending = list(enumerate(occupancies))
    while pending:
        block, pending = pending[:4], pending[4:]
        requests = [env.build_request(o) for _, o in block]
        results = relax_batch(physics, cache, namespace, requests, ledger, log=log,
                              meta=[dict(kind='initial', start=index) for index, _ in block])
        for (index, occupancy), result in zip(block, results):
            if isinstance(result, Exception):
                if not spares: raise RuntimeError('Initial state failed and no spare start remains')
                replaced.append(dict(start=index, reason=type(result).__name__))
                pending.append((index, spares.pop(0)))
            else:
                terminals.append((index, result))
    ledger.replaced_starts = getattr(ledger, 'replaced_starts', []) + replaced
    return [t for _, t in sorted(terminals, key=lambda x: x[0])]


def step_chains(chains, physics, cache, ledger, *, step, log=None):
    """Advance up to four chains by one synchronized step."""
    proposals = [chain.propose() for chain in chains]
    requests, prompts = [], []
    for chain, (action, _) in zip(chains, proposals):
        prompts.append(env.encode_prompt_v2(chain.state()))
        requests.append(env.build_request(env.apply_action(chain.current.mapping.occupancy, action)))
    ledger.counters.attempts += len(chains)
    ledger.counters.physical_action_attempts += len(chains)
    results = relax_batch(physics, cache, chains[0].namespace, requests, ledger, log=log,
                          meta=[dict(kind='step', chain=c.chain_id, step=step) for c in chains])
    for chain, (action, uniform), prompt, result in zip(chains, proposals, prompts, results):
        text = action.to_json()
        if isinstance(result, Exception):
            status = 'UNCONVERGED' if isinstance(result, Unconverged) else 'UNMAPPABLE_TERMINAL'
            if status == 'UNMAPPABLE_TERMINAL': ledger.counters.unmappable_terminals += 1
            chain.history.append(env.HistoryEntry(text, status, None, None, None, 0.))
            chain.steps.append(dict(step=step, action=text, status=status, accepted=False))
            continue
        delta = result.result.energy - chain.current.result.energy
        accept = bool(chain.mode == 'random' or delta <= 0 or uniform < np.exp(-delta / chain.temperatures[step]))
        key = result.mapping.canonical_key
        row = dict(step=step, action=text, status='success' if accept else 'rejected', accepted=accept,
                   energy_delta_eV=float(delta), energy_eV=float(result.result.energy), terminal_key=key,
                   near=bool(result.mapping.quality_metadata.get('NEAR_THRESHOLD', False)))
        chain.steps.append(row)
        chain.history.append(env.HistoryEntry(text, row['status'], delta, None, key, 0.))
        if accept:
            if delta < 0:
                chain.rows.append(dict(messages=prompt, action=json.loads(text), chain=chain.chain_id,
                                       start=chain.start_id, step=step, energy_delta_eV=delta))
            chain.current = result; chain.best = min(chain.best, result.result.energy); chain.visits.add(key)
    if env.unmappable_rate_exceeded(ledger.counters.physical_action_attempts, ledger.counters.unmappable_terminals):
        raise env.UnmappableRateExceeded(ledger.counters)


def run_chains(modes_and_starts, physics, cache, namespace, ledger, *, rng_label, seed, log=None, chain_offset=0,
               temperatures=TEMPERATURES):
    """Run chains in synchronized batches of four; each is K steps from its start terminal."""
    rng = stream(seed, rng_label)
    chains = [Chain(mode, terminal, chain_id=chain_offset + i, start_id=start_id,
                    rng=np.random.default_rng(int(rng.integers(0, 2**63 - 1))), namespace=namespace, ledger=ledger,
                    temperatures=temperatures)
              for i, (mode, start_id, terminal) in enumerate(modes_and_starts)]
    for begin in range(0, len(chains), 4):
        block = chains[begin:begin + 4]
        for step in range(K):
            ledger.check()
            step_chains(block, physics, cache, ledger, step=step, log=log)
    return chains


def chain_summary(chain):
    return dict(chain=chain.chain_id, start=chain.start_id, mode=chain.mode, best_energy_eV=chain.best,
                initial_energy_eV=None, steps=chain.steps, visits=sorted(chain.visits),
                sft_rows=len(chain.rows))


def collect_d0(physics, cache, namespace, starts, ledger, *, seed, log=None, spares=()):
    """52 chains (26 SA, 26 random) from 52 distinct starts: 52 + 416 = 468 relaxations."""
    if len(starts) != len(D0_MIX): raise ValueError('D0 start pool size changed')
    terminals = relax_initials(physics, cache, namespace, starts, ledger, log=log, spares=spares)
    modes = [('SA' if i % 2 == 0 else 'random') for i in range(len(starts))]
    chains = run_chains([(m, i, t) for i, (m, t) in enumerate(zip(modes, terminals))], physics, cache,
                        namespace, ledger, rng_label='d0', seed=seed, log=log)
    initial = {i: t.result.energy for i, t in enumerate(terminals)}
    summaries = []
    for chain in chains:
        row = chain_summary(chain); row['initial_energy_eV'] = initial[chain.start_id]; summaries.append(row)
    return chains, summaries, terminals


def run_search_round(physics, cache, namespace, starts, ledger, *, seed, label, log=None, chains_per_start=4, spares=(),
                     temperatures=TEMPERATURES):
    """SA baseline with the LLM group shape: per start, 1 initial + 4 SA chains x K."""
    terminals = relax_initials(physics, cache, namespace, starts, ledger, log=log, spares=spares)
    specs = [('SA', s, terminals[s]) for s in range(len(starts)) for _ in range(chains_per_start)]
    chains = run_chains(specs, physics, cache, namespace, ledger, rng_label='search-' + label, seed=seed, log=log,
                        temperatures=temperatures)
    out = []
    for chain in chains:
        row = chain_summary(chain); row['initial_energy_eV'] = terminals[chain.start_id].result.energy; out.append(row)
    return out, terminals
