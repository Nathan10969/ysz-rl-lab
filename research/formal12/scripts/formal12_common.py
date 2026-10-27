"""Review12 formal experiment: shared constants, start pools, ledgers, physics.

Environment v2.2 (claude08_env), batch MACE physics (claude10_physics) and the
Review11 constrained actor are reused unchanged. Everything here is CPU-only
on import.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import claude08_env as env
from calibration import make_state

SCHEMA = 'review12-formal-v1'
SEEDS = (2026102301, 2026102302, 2026102303)
ARMS = ('frozen', 'itersft', 'grpo_traj', 'grpo_step')
SEARCH_ARM = 'search'
ROUNDS = 3
GROUPS_PER_ROUND = 16
GROUP_SIZE = 4
K = 8
UPDATE_EVERY_GROUPS = 4
D0_MIX = ('SA',) * 24 + ('random',) * 24 + ('GA',) * 4   # 52 trajectories, 512 action relaxations
D0_RELAX_CAP = 52 + 24 * 8 + 24 * 8 + 4 * 32               # 564
ROUND_RELAX_CAP = GROUPS_PER_ROUND * (1 + GROUP_SIZE * K)  # 528
BLIND_STARTS = 2
BLIND_RELAX_CAP = BLIND_STARTS * (1 + GROUP_SIZE * K)      # 66 (<= 72 budget)
ARM_RELAX_BUDGET = D0_RELAX_CAP + ROUNDS * ROUND_RELAX_CAP + 72   # 2220 per arm/seed incl. shared D0
INTERNAL_SECONDS = 6600
SPARE_STARTS = 8
# Review12.1: a relaxation that reaches 300 LBFGS steps unconverged is paid, rolled back, reward 0,
# counted separately (as UNMAPPABLE, v2.1). Stop if it becomes systematic.
UNCONVERGED_STOP_MIN, UNCONVERGED_STOP_FRACTION = 5, .02
# Token caps per job (card-hours are bounded by Slurm, not by these).
LIMITS = env.BudgetLimits(max_attempts=8, max_relax_actual=ROUND_RELAX_CAP, max_raw_ef=200000,
                          max_generated_tokens=40000, max_prefill_tokens=12000000, max_seen_tokens=12000000,
                          wall_seconds=INTERNAL_SECONDS)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (set, frozenset)): return sorted(value)
    raise TypeError('Not JSON serializable: ' + type(value).__name__)


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=1, allow_nan=False, default=_plain) + '\n')
    temporary.replace(path)


def append_jsonl(path, row):
    if path is not None:
        with Path(path).open('a') as handle: handle.write(json.dumps(row, allow_nan=False, default=_plain) + '\n')


def source_hashes(root=None):
    root = Path(__file__).resolve().parent if root is None else Path(root)
    return {p.name: sha256(p) for p in sorted(root.glob('*.py'))}


def stream(seed, label):
    """Independent named RNG streams per seed (teacher, starts, rollout, ...)."""
    words = [seed] + list(hashlib.sha256(label.encode()).digest()[:8])
    return np.random.default_rng(np.random.SeedSequence(words))


def start_pools(seed):
    """Disjoint canonical start families: D0, rounds 1..3, blind, held-out."""
    rng = stream(seed, 'starts')
    sizes = dict(d0=len(D0_MIX), **{f'round{r}': GROUPS_PER_ROUND for r in range(1, ROUNDS + 1)},
                 blind=BLIND_STARTS, spare=SPARE_STARTS,   # spares are drawn after the original pools
                 # Review12.3 (drawn after all original pools, which therefore stay unchanged):
                 val=8, **{f'tune_{i}': GROUPS_PER_ROUND for i in range(1, 9)},   # meta-layer validation / training starts
                 round4=GROUPS_PER_ROUND, round5=GROUPS_PER_ROUND)                # extension rounds
    total = sum(sizes.values())
    keys, occupancies = set(), []
    while len(occupancies) < total:
        occupancy = env.validate_occupancy(make_state(int(rng.integers(0, 2**63 - 1)), 3, 16).numbers)
        key = env.occupancy_key(occupancy)
        if key in keys: continue
        keys.add(key); occupancies.append(list(occupancy))
    pools, offset = {}, 0
    for name, size in sizes.items():
        pools[name] = occupancies[offset:offset + size]; offset += size
    return pools


class Ledger:
    """Job counters plus wall clock, shared by physics and token charges."""

    def __init__(self, limits=LIMITS, *, started_at=None, clock=time.monotonic, log_path=None):
        self.limits, self.clock = limits, clock
        self.counters = env.Counters()
        self.started_at = clock() if started_at is None else started_at
        self.raw = env.RawEFCounter(self.counters, limits, started_at=self.started_at, clock=clock)
        self.events, self.phase_timings, self.log_path = [], [], log_path
        self.unconverged = 0

    def note_unconverged(self):
        self.unconverged += 1
        attempts = max(1, self.counters.physical_action_attempts)
        if self.unconverged >= UNCONVERGED_STOP_MIN and self.unconverged > UNCONVERGED_STOP_FRACTION * attempts:
            raise RuntimeError(f'Systematic unconverged relaxations: {self.unconverged}/{attempts}')

    def elapsed(self):
        return self.clock() - self.started_at

    def check(self):
        # Physics caps (relax_actual, raw_ef) are enforced where physics is charged (charge_relax,
        # RawEFCounter.before_forward, Environment._charge_attempt); a job that used exactly its relaxation
        # cap may still finish its non-physical work (update, save). Review12.2 fix after job 752.
        self.raw.check_time()
        for name in ('relax_actual', 'raw_ef'):
            bound = getattr(self.limits, 'max_' + name)
            if bound is not None and getattr(self.counters, name) > bound:
                raise env.BudgetExceeded(name + ' hard cap exceeded')
        for name in ('generated_tokens', 'prefill_tokens', 'seen_tokens'):
            bound = getattr(self.limits, 'max_' + name)
            if bound is not None and getattr(self.counters, name) >= bound:
                raise env.BudgetExceeded(name + ' hard cap reached')

    def _charge(self, delta, phase):
        self.check()
        for name, n in delta.items():
            bound = getattr(self.limits, 'max_' + name)
            if bound is not None and getattr(self.counters, name) + n > bound:
                raise env.BudgetExceeded(name + ' cap before ' + phase)
        for name, n in delta.items(): setattr(self.counters, name, getattr(self.counters, name) + n)

    def before_prefill(self, n, *, phase): self._charge({'prefill_tokens': n, 'seen_tokens': n}, phase)
    def before_generated_token(self, *, phase): self._charge({'generated_tokens': 1, 'seen_tokens': 1}, phase)

    def charge_relax(self, n):
        self.check()
        if self.limits.max_relax_actual is not None and self.counters.relax_actual + n > self.limits.max_relax_actual:
            raise env.BudgetExceeded('Actual relaxation cap')
        self.counters.relax_actual += n


class Phase:
    """Context manager recording wall time per phase (synchronizes CUDA)."""

    def __init__(self, ledger, name, **metadata):
        self.ledger, self.name, self.metadata = ledger, name, metadata

    def __enter__(self):
        self.begin = self.ledger.clock(); return self

    def __exit__(self, kind, value, traceback):
        import sys
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized(): torch.cuda.synchronize()
        self.ledger.phase_timings.append(dict(phase=self.name, seconds=self.ledger.clock() - self.begin,
            end_elapsed_seconds=self.ledger.elapsed(), error=None if value is None else repr(value), **self.metadata))
        return False


def verify_physics_lanes(requests, values, ledger_before_raw, ledger):
    """Formal v2.2 LBFGS receipt per lane; raw E/F reconciles with the counter."""
    total = 0
    for request, value in zip(requests, values):
        evidence = dict(getattr(value, 'evidence', {}))
        raw = evidence.get('raw_ef_attempts')
        formal = evidence.get('formal_receipt') is True and evidence.get('status') == 'converged'
        capped = (evidence.get('formal_receipt') is False and evidence.get('status') == 'step_capped_unconverged'
                  and value.converged is False)
        if ((not formal and not capped) or type(raw) is not int or raw < 1 or evidence.get('optimizer') != 'LBFGS'
                or evidence.get('protocol') != env.PROTOCOL or evidence.get('protocol_sha') != request.protocol_sha
                or evidence.get('perturbation_seed') != request.perturbation_seed
                or evidence.get('relaxation_spec') != asdict(request.spec)):
            raise RuntimeError('Formal v2.2 LBFGS receipt required per lane')
        total += raw
    if ledger.counters.raw_ef - ledger_before_raw != total:
        raise RuntimeError('Raw E/F cost does not reconcile with lane receipts')


def runtime_info(device):
    if device != 'cuda':
        return dict(device=device, GPU_work_performed=False)
    from smoke_grpo08 import require_single_gpu_job
    info = require_single_gpu_job()
    if 'A100' not in info.get('device_name', ''): raise RuntimeError('Review12 requires one Slurm CUDA A100')
    return dict(info, device='cuda', GPU_work_performed=True)


def make_physics(checkpoint, output, device, actor=None):
    if os.environ.get('REVIEW12_SYNTHETIC_PHYSICS') == '1':
        if device != 'cpu': raise RuntimeError('Synthetic physics is CPU-test only')
        return SyntheticPhysics()
    from claude10_physics import BatchMACEPhysics
    return BatchMACEPhysics(checkpoint, Path(output) / 'physics_engine', device=device, actor=actor)


class SyntheticPhysics:
    """CPU-test stand-in: request geometry back, energy from Y-vacancy shells. Never evidence."""

    def __init__(self):
        self.device = 'cpu'
        self.last_evidence = []

    def relax_many(self, requests, counter, max_steps=300):
        from calibration import reference, shell_counts
        from claude10_physics import BatchPhysicsResult
        import torch
        torch.set_default_dtype(torch.float64)   # mimic BatchMACEPhysics' global side effect
        counter.before_forward(len(requests))
        out, self.last_evidence = [], []
        for i, r in enumerate(requests):
            state = reference(3); state.numbers[:] = r.occupancy
            shells = shell_counts(state)
            noise = int(hashlib.sha256(r.key.encode()).hexdigest()[:6], 16) / 16**6
            energy = -3000. + .15 * shells['Y_vac_NN'] - .05 * shells['Y_vac_NNN'] + .1 * noise
            every = int(os.environ.get('REVIEW12_SYNTHETIC_UNCONVERGED_EVERY', '0'))
            capped = every > 0 and int(hashlib.sha256(r.key.encode()).hexdigest()[:8], 16) % every == 0
            evidence = dict(formal_receipt=not capped, status='step_capped_unconverged' if capped else 'converged',
                            raw_ef_attempts=1, nsteps=300 if capped else 0, lane=i,
                            optimizer='LBFGS', protocol=env.PROTOCOL, protocol_sha=r.protocol_sha,
                            perturbation_seed=r.perturbation_seed, relaxation_spec=asdict(r.spec))
            self.last_evidence.append(evidence)
            forces = np.full((len(r.atoms), 3), .1) if capped else np.zeros((len(r.atoms), 3))
            out.append(BatchPhysicsResult(r.atoms.copy(), energy, forces, not capped, evidence=evidence))
        return out

    def close(self):
        pass
