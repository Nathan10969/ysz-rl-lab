"""CPU state machine for Review08; importing this module never loads a model.

The only physical boundary is ``callback(request, raw_ef_counter)``. A caller
must call ``raw_ef_counter.before_forward(batch_size)`` before every real model
forward, including warmup, line searches and failed forwards. Returned energies
are eV/cell and forces are eV/Angstrom. No real callback is supplied here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import hashlib
import json
import time
from typing import Callable

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from scipy.optimize import linear_sum_assignment

from calibration import canonical_key, reference, shell_counts


PROTOCOL = "YSZ08_terminal_local2_v1"
MODEL_SHA = "d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9"
SECOND_SHELL = {"swap_cations": 5.14, "move_vacancy": 3.63452886}
NEAREST_NEIGHBOR = {"cations": 5.14 / np.sqrt(2), "oxygen": 2.57}
ACTION_TOLERANCE = 1e-6
AMBIGUITY_TOLERANCE = 1e-8
COMPOSITION = {0: 8, 8: 208, 39: 16, 40: 92}


class ActionError(ValueError):
    """A policy output cannot be executed under the fixed action protocol."""


class MappingError(ValueError):
    """A relaxed structure has no sufficiently certain parent occupation."""


class BudgetExceeded(RuntimeError):
    """The next operation would cross a hard bound; it must not execute."""


@lru_cache(maxsize=1)
def _parent():
    return reference(3)


@lru_cache(maxsize=1)
def _distances():
    return _parent().get_all_distances(mic=True)


def validate_occupancy(occupancy) -> tuple[int, ...]:
    """Validate numbered mother sites, including their original sublattices."""
    values = np.asarray(occupancy)
    if values.shape != (324,) or values.dtype.kind not in "iu":
        raise ValueError("Occupation must contain exactly 324 integer element codes")
    counts = dict(zip(*np.unique(values, return_counts=True)))
    if counts != COMPOSITION:
        raise ValueError("Expected Zr92Y16O208 with eight oxygen vacancies")
    cats = _parent().numbers == 40
    if not np.all(np.isin(values[cats], [39, 40])) or not np.all(np.isin(values[~cats], [0, 8])):
        raise ValueError("Species cannot cross the mother cation/anion sublattices")
    return tuple(map(int, values))


def occupancy_key(occupancy) -> str:
    state = _parent().copy()
    state.numbers[:] = validate_occupancy(occupancy)
    return canonical_key(state, 3)


@dataclass(frozen=True)
class Action:
    op: str
    sites: tuple[int, int]

    def to_json(self) -> str:
        return json.dumps({"op": self.op, "sites": list(self.sites)}, separators=(",", ":"))


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ActionError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_action(text: str) -> Action:
    """Accept a single JSON object; Python bool is deliberately not an integer."""
    if not isinstance(text, str):
        raise ActionError("Policy action must be JSON text")
    try:
        obj = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except (ValueError, TypeError) as exc:
        raise ActionError(str(exc)) from exc
    if not isinstance(obj, dict) or set(obj) != {"op", "sites"}:
        raise ActionError("Exactly op and sites are required")
    if not isinstance(obj["op"], str) or obj["op"] not in SECOND_SHELL:
        raise ActionError("Unknown action operation")
    sites = obj["sites"]
    if not isinstance(sites, list) or len(sites) != 2 or any(type(i) is not int for i in sites):
        raise ActionError("sites must contain two JSON integers")
    if sites[0] == sites[1] or min(sites) < 0 or max(sites) >= 324:
        raise ActionError("sites must be distinct indices in 0..323")
    return Action(obj["op"], tuple(sites))


def apply_action(occupancy, action: Action) -> tuple[int, ...]:
    values = np.asarray(validate_occupancy(occupancy), dtype=int)
    # Validate programmatic Actions under exactly the same rules as policy text.
    try:
        action = parse_action(action.to_json())
    except (AttributeError, TypeError, ValueError) as exc:
        raise ActionError(str(exc)) from exc
    i, j = action.sites
    parent_numbers = _parent().numbers
    expected = {39, 40} if action.op == "swap_cations" else {0, 8}
    expected_parent = 40 if action.op == "swap_cations" else 8
    if parent_numbers[i] != expected_parent or parent_numbers[j] != expected_parent:
        raise ActionError("Action must stay on its mother sublattice")
    if {int(values[i]), int(values[j])} != expected:
        raise ActionError("Action endpoints must be Y/Zr or vacancy/O")
    if _distances()[i, j] > SECOND_SHELL[action.op] + ACTION_TOLERANCE:
        raise ActionError("Action exceeds the parent second-neighbor shell")
    values[i], values[j] = values[j], values[i]
    return validate_occupancy(values)


def enumerate_legal_actions(occupancy) -> tuple[Action, ...]:
    """Stable numbered action order; no model evaluation or material filter."""
    values = np.asarray(validate_occupancy(occupancy))
    actions = []
    for op, left, right in [("swap_cations", 39, 40), ("move_vacancy", 0, 8)]:
        for i in np.flatnonzero(values == left):
            for j in np.flatnonzero(values == right):
                if _distances()[i, j] <= SECOND_SHELL[op] + ACTION_TOLERANCE:
                    actions.append(Action(op, tuple(sorted((int(i), int(j))))))
    return tuple(sorted(actions, key=lambda action: (action.op, action.sites)))


def sample_legal_actions(occupancy, count: int, *, seed: int) -> tuple[Action, ...]:
    """CPU candidate helper only; this does not generate or score a D0 dataset."""
    actions = enumerate_legal_actions(occupancy)
    if type(count) is not int or not 0 <= count <= len(actions):
        raise ValueError("Requested count must fit the legal set without replacement")
    indices = np.random.default_rng(seed).choice(len(actions), count, replace=False)
    return tuple(actions[int(i)] for i in indices)


def ideal_atoms(occupancy) -> Atoms:
    """Reconstruct the next request at ideal mother coordinates, omitting 0."""
    values = np.asarray(validate_occupancy(occupancy))
    atoms = _parent().copy()
    atoms.numbers[:] = values
    return atoms[values != 0]


@dataclass(frozen=True)
class RelaxationSpec:
    model_sha: str = MODEL_SHA
    dtype: str = "float64"
    optimizer: str = "BFGS"
    fmax: float = .03
    steps: int = 300
    maxstep: float = .1

    def __post_init__(self):
        if self.optimizer != "BFGS" or self.fmax != .03 or self.steps != 300 or self.maxstep != .1:
            raise ValueError("Review08 fixes BFGS/.03/300/maxstep .1")
        if not isinstance(self.model_sha, str) or not self.model_sha:
            raise ValueError("A model SHA is required")
        if type(self.dtype) is not str or self.dtype != "float64":
            raise ValueError("Review08 requires exactly float64")


def _hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PhysicsRequest:
    occupancy: tuple[int, ...]
    atoms: Atoms
    spec: RelaxationSpec
    key: str


def build_request(occupancy, spec: RelaxationSpec | None = None) -> PhysicsRequest:
    spec = RelaxationSpec() if spec is None else spec
    values = validate_occupancy(occupancy)
    atoms = ideal_atoms(values)
    payload = {"protocol": PROTOCOL, "occupancy": values, "composition": COMPOSITION,
               "cell": atoms.cell.array.tolist(), "pbc": atoms.pbc.tolist(),
               "initial_numbers": atoms.numbers.tolist(), "initial_positions": atoms.positions.tolist(),
               "relaxation": asdict(spec)}
    return PhysicsRequest(values, atoms, spec, _hash_json(payload))


@dataclass
class PhysicsResult:
    atoms: Atoms
    energy: float
    forces: np.ndarray
    converged: bool


@dataclass(frozen=True)
class TerminalMapping:
    occupancy: tuple[int, ...]
    canonical_key: str
    assigned_site_ids: tuple[int, ...]
    assignment_distances: tuple[float, ...]
    geometry_hash: str
    shell_counts: dict[str, int]


def assignment_occupation_ambiguous(cost, species, assignment, tolerance=AMBIGUITY_TOLERANCE) -> bool:
    """Test equal-cost alternative occupations, ignoring identical-species swaps.

    Pad an oxygen assignment with vacancy rows. Every alternative assignment
    decomposes into alternating cycles. A different occupation requires a cycle
    edge between unlike row species. The shortest return path gives the cheapest
    complete cycle for each such edge. This also handles nonlocal degeneracy.
    ``cost`` is squared MIC distance, not distance.
    """
    cost = np.asarray(cost, dtype=float)
    species = np.asarray(species, dtype=int)
    assignment = np.asarray(assignment, dtype=int)
    if cost.ndim != 2 or cost.shape[0] > cost.shape[1] or len(species) != cost.shape[0]:
        raise ValueError("Invalid assignment cost/species shape")
    rows, columns = cost.shape
    if assignment.shape != (rows,) or len(set(assignment.tolist())) != rows:
        raise ValueError("Assignment is not one-to-one")
    if not np.all(np.isfinite(cost)) or np.any(assignment < 0) or np.any(assignment >= columns):
        raise ValueError("Invalid assignment costs or indices")
    unused = np.setdiff1d(np.arange(columns), assignment)
    padded_cost = np.vstack((cost, np.zeros((columns - rows, columns))))
    padded_species = np.concatenate((species, np.zeros(columns - rows, dtype=int)))
    assigned = np.concatenate((assignment, unused))
    # Edge r->s means row r takes the site currently occupied by row s.
    edge = padded_cost[:, assigned] - padded_cost[np.arange(columns), assigned][:, None]
    shortest = edge.copy()
    np.fill_diagonal(shortest, 0)
    for k in range(columns):
        shortest = np.minimum(shortest, shortest[:, k, None] + shortest[None, k, :])
    differing_species = padded_species[:, None] != padded_species[None, :]
    return bool(np.any((edge + shortest.T <= tolerance) & differing_species))


def map_terminal(atoms: Atoms) -> TerminalMapping:
    """Global periodic squared-distance assignment; site_id is never read."""
    parent = _parent()
    if len(atoms) != 316 or dict(zip(*np.unique(atoms.numbers, return_counts=True))) != {8: 208, 39: 16, 40: 92}:
        raise MappingError("Relaxed physical composition must be Zr92Y16O208")
    if not np.all(atoms.pbc) or not np.allclose(atoms.cell.array, parent.cell.array, rtol=0, atol=1e-10):
        raise MappingError("Fixed periodic mother cell changed")
    if not np.all(np.isfinite(atoms.positions)):
        raise MappingError("Nonfinite relaxed positions")
    assigned = np.full(len(atoms), -1, dtype=int)
    distances = np.full(len(atoms), np.nan)
    for mask, sites, name in [(np.isin(atoms.numbers, [39, 40]), np.flatnonzero(parent.numbers == 40), "cations"),
                              (atoms.numbers == 8, np.flatnonzero(parent.numbers == 8), "oxygen")]:
        indices = np.flatnonzero(mask)
        delta = atoms.positions[indices, None] - parent.positions[sites][None]
        mic = find_mic(delta.reshape(-1, 3), parent.cell, pbc=True)[1].reshape(len(indices), len(sites))
        cost = mic ** 2
        row, col = linear_sum_assignment(cost)
        if len(row) != len(indices):
            raise MappingError("Incomplete terminal assignment")
        assigned[indices[row]] = sites[col]
        distances[indices[row]] = mic[row, col]
        if np.max(mic[row, col]) >= .5 * NEAREST_NEIGHBOR[name]:
            raise MappingError("Terminal distance reaches half the sublattice nearest neighbor")
        ordered_col = np.empty(len(row), dtype=int)
        ordered_col[row] = col
        if assignment_occupation_ambiguous(cost, atoms.numbers[indices], ordered_col):
            raise MappingError("Equal-cost terminal assignments give different occupations")
    if np.any(assigned < 0) or len(np.unique(assigned)) != 316:
        raise MappingError("Terminal mapping is not one-to-one")
    values = np.zeros(324, dtype=int)
    values[assigned] = atoms.numbers
    occupancy = validate_occupancy(values)
    mapped = parent.copy()
    mapped.numbers[:] = values
    # Atom-order-independent geometry identity, retaining numbered sites and R.
    order = np.argsort(assigned)
    fractional = atoms.get_scaled_positions(wrap=True)[order]
    fractional = np.mod(np.round(fractional, 12), 1)
    geometry_hash = _hash_json({"cell": atoms.cell.array.tolist(), "sites": assigned[order].tolist(),
                                "numbers": atoms.numbers[order].tolist(), "fractional_positions": fractional.tolist()})
    return TerminalMapping(occupancy, canonical_key(mapped, 3), tuple(map(int, assigned)),
                           tuple(map(float, distances)), geometry_hash, shell_counts(mapped))


@dataclass
class VerifiedTerminal:
    result: PhysicsResult
    mapping: TerminalMapping

    def copy(self):
        mapping = replace(self.mapping, shell_counts=dict(self.mapping.shell_counts))
        return VerifiedTerminal(PhysicsResult(self.result.atoms.copy(), self.result.energy,
                                             self.result.forces.copy(), self.result.converged), mapping)


def verify_result(result: PhysicsResult, spec: RelaxationSpec) -> VerifiedTerminal:
    if not isinstance(result, PhysicsResult):
        raise ValueError("Callback must return PhysicsResult")
    forces = np.asarray(result.forces, dtype=float)
    if forces.shape != (len(result.atoms), 3) or not np.all(np.isfinite(forces)):
        raise ValueError("Finite per-atom Cartesian forces are required")
    if type(result.energy) is bool or not np.isscalar(result.energy) or not np.isfinite(result.energy):
        raise ValueError("Finite energy is required")
    if result.converged is not True or np.max(np.linalg.norm(forces, axis=1), initial=0) > spec.fmax:
        raise ValueError("Relaxation did not converge to fmax <= .03")
    mapping = map_terminal(result.atoms)
    return VerifiedTerminal(PhysicsResult(result.atoms.copy(), float(result.energy), forces.copy(), True), mapping)


class RequestCache:
    """Exact numbered requests, isolated by (arm, seed), verified on insertion."""
    def __init__(self):
        self._requests = {}
        self._terminals = {}
        self.diagnostics = []

    def get(self, namespace, request: PhysicsRequest) -> VerifiedTerminal | None:
        entry = self._requests.get((tuple(namespace), request.key))
        return None if entry is None else entry.copy()

    def record(self, namespace, request: PhysicsRequest, result: PhysicsResult) -> VerifiedTerminal:
        # Public insertion verifies independently; callers cannot label arbitrary
        # callback output as trusted or use a terminal key as a request key.
        terminal = verify_result(result, request.spec)
        namespace = tuple(namespace)
        identity = (namespace, terminal.mapping.canonical_key)
        records = self._terminals.setdefault(identity, [])
        energies = [record.result.energy for record in records] + [terminal.result.energy]
        if max(energies) - min(energies) > .01:
            self.diagnostics.append({"kind": "terminal_energy_conflict", "namespace": namespace,
                                     "terminal_key": identity[1], "energy_span_eV": max(energies) - min(energies)})
        if records and any(record.mapping.geometry_hash != terminal.mapping.geometry_hash for record in records):
            self.diagnostics.append({"kind": "terminal_geometry_variants", "namespace": namespace,
                                     "terminal_key": identity[1], "geometry_hash": terminal.mapping.geometry_hash})
        records.append(terminal.copy())
        self._requests[(namespace, request.key)] = terminal.copy()
        return terminal

    def terminal_records(self, namespace, terminal_key) -> tuple[VerifiedTerminal, ...]:
        return tuple(record.copy() for record in self._terminals.get((tuple(namespace), terminal_key), []))


@dataclass(frozen=True)
class BudgetLimits:
    max_attempts: int = 8
    max_relax_actual: int | None = None
    max_raw_ef: int | None = None
    max_generated_tokens: int | None = None
    max_prefill_tokens: int | None = None
    max_seen_tokens: int | None = None
    wall_seconds: float | None = None

    def __post_init__(self):
        if type(self.max_attempts) is not int or not 0 <= self.max_attempts <= 8:
            raise ValueError("Review08 allows at most eight attempts per episode")
        for name, value in asdict(self).items():
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError("Budget limits must be finite and nonnegative: " + name)
            if name != "wall_seconds" and value is not None and type(value) is not int:
                raise ValueError("Count limits must be integers: " + name)


@dataclass
class Counters:
    """Cost ledger and rollout novelty events, not a global diversity ledger.

    With shared counters, one key newly visited in each of four private G4
    rollouts counts as four novelty_events. Global N_unique is derived from
    the merged terminal-key set, with initial keys handled explicitly.
    """
    attempts: int = 0
    relax_actual: int = 0
    raw_ef: int = 0
    forward_calls: int = 0
    cache_hits: int = 0
    novelty_events: int = 0
    generated_tokens: int = 0
    prefill_tokens: int = 0
    seen_tokens: int = 0


class RawEFCounter:
    def __init__(self, counters: Counters, limits: BudgetLimits, *, started_at=None, clock=time.monotonic):
        self.counters, self.limits, self.clock = counters, limits, clock
        self.started_at = clock() if started_at is None else started_at

    def check_time(self):
        if self.limits.wall_seconds is not None and self.clock() - self.started_at >= self.limits.wall_seconds:
            raise BudgetExceeded("Wall-time cap reached before next operation")

    def before_forward(self, batch_size=1):
        """Charge batch b as b structures *before* executing the real forward."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Forward batch_size must be a positive integer")
        self.check_time()
        if self.limits.max_raw_ef is not None and self.counters.raw_ef + batch_size > self.limits.max_raw_ef:
            raise BudgetExceeded("Raw E/F cap reached before next forward")
        self.counters.raw_ef += batch_size
        self.counters.forward_calls += 1


@dataclass(frozen=True)
class HistoryEntry:
    action: str
    status: str
    energy_delta: float | None
    request_key: str | None
    terminal_key: str | None
    reward: float


@dataclass
class EpisodeState:
    occupancy: tuple[int, ...]
    geometry: Atoms
    energy: float
    best_energy: float
    terminal_key: str
    geometry_hash: str
    frozen_history: frozenset[str]
    private_visits: set[str]
    limits: BudgetLimits
    counters: Counters
    namespace: tuple[str, int]
    started_at: float
    clock: Callable[[], float] = field(repr=False, compare=False)
    steps_taken: int = 0
    history: list[HistoryEntry] = field(default_factory=list)


@dataclass(frozen=True)
class StepResult:
    success: bool
    status: str
    reward: float
    energy_reward: float
    novelty_reward: float
    cache_hit: bool = False
    error: str | None = None
    request_key: str | None = None
    terminal_key: str | None = None


class Environment:
    def __init__(self, initial: PhysicsResult, callback: Callable, *, arm: str, seed: int,
                 spec=None, limits=None, counters=None, cache=None, frozen_history=frozenset(),
                 started_at=None, clock=time.monotonic):
        self.spec = RelaxationSpec() if spec is None else spec
        self.limits = BudgetLimits() if limits is None else limits
        self.counters = Counters() if counters is None else counters
        self.cache = RequestCache() if cache is None else cache
        self.namespace = (arm, seed)
        self.callback = callback
        self.raw_counter = RawEFCounter(self.counters, self.limits, started_at=started_at, clock=clock)
        terminal = verify_result(initial, self.spec)
        visits = set(frozen_history) | {terminal.mapping.canonical_key}
        self.state = EpisodeState(terminal.mapping.occupancy, terminal.result.atoms.copy(), terminal.result.energy,
                                  terminal.result.energy, terminal.mapping.canonical_key, terminal.mapping.geometry_hash,
                                  frozenset(frozen_history), visits, self.limits, self.counters, self.namespace,
                                  self.raw_counter.started_at, self.raw_counter.clock)

    def _charge_attempt(self, generated_tokens, prefill_tokens):
        self.raw_counter.check_time()
        # Exhausted hard bounds stop even free cache/inference steps. Otherwise
        # a caller could extend a run indefinitely through already known states.
        for name in ("relax_actual", "raw_ef", "generated_tokens", "prefill_tokens", "seen_tokens"):
            bound = getattr(self.limits, "max_" + name)
            if bound is not None and getattr(self.counters, name) >= bound:
                raise BudgetExceeded(name + " cap reached before next attempt")
        if any(type(value) is not int or value < 0 for value in (generated_tokens, prefill_tokens)):
            raise ValueError("Token charges must be nonnegative integers")
        if self.state.steps_taken >= self.limits.max_attempts:
            raise BudgetExceeded("Episode attempt cap reached before next attempt")
        proposed = {"attempts": self.counters.attempts + 1,
                    "generated_tokens": self.counters.generated_tokens + generated_tokens,
                    "prefill_tokens": self.counters.prefill_tokens + prefill_tokens,
                    "seen_tokens": self.counters.seen_tokens + generated_tokens + prefill_tokens}
        for name, value in proposed.items():
            if name == "attempts":
                continue  # Attempts are bounded per episode; physics/tokens can share a ledger.
            bound = getattr(self.limits, "max_" + name)
            if bound is not None and value > bound:
                raise BudgetExceeded(name + " cap reached before next attempt")
        for name, value in proposed.items():
            setattr(self.counters, name, value)
        self.state.steps_taken += 1

    def _failure(self, text, status, error, request_key=None):
        self.state.history.append(HistoryEntry(text, status, None, request_key, None, -1.0))
        return StepResult(False, status, -1.0, 0.0, 0.0, error=str(error), request_key=request_key)

    def step(self, text: str, *, generated_tokens=0, prefill_tokens=0) -> StepResult:
        self._charge_attempt(generated_tokens, prefill_tokens)
        try:
            occupancy = apply_action(self.state.occupancy, parse_action(text))
        except ActionError as exc:
            return self._failure(text, "illegal", exc)
        request = build_request(occupancy, self.spec)
        terminal = self.cache.get(self.namespace, request)
        cache_hit = terminal is not None
        if cache_hit:
            self.counters.cache_hits += 1
        else:
            self.raw_counter.check_time()
            if self.limits.max_relax_actual is not None and self.counters.relax_actual >= self.limits.max_relax_actual:
                raise BudgetExceeded("Actual relaxation cap reached before callback")
            # A callback exception, bad force or failed mapping never refunds it.
            self.counters.relax_actual += 1
            try:
                result = self.callback(request, self.raw_counter)
                terminal = self.cache.record(self.namespace, request, result)
            except BudgetExceeded as exc:
                self._failure(text, "budget_exhausted", exc, request.key)
                raise  # A hard stop cannot become another trainable retry.
            except Exception as exc:
                return self._failure(text, "physical_failure", exc, request.key)
        old_energy = self.state.energy
        best = min(self.state.best_energy, terminal.result.energy)
        energy_reward = max(0.0, self.state.best_energy - best) / .20
        new_terminal = terminal.mapping.canonical_key not in self.state.private_visits
        novelty_reward = .05 if new_terminal else 0.0
        self.state.private_visits.add(terminal.mapping.canonical_key)
        if new_terminal:
            self.counters.novelty_events += 1
        self.state.occupancy = terminal.mapping.occupancy
        self.state.geometry = terminal.result.atoms.copy()
        self.state.energy = terminal.result.energy
        self.state.best_energy = best
        self.state.terminal_key = terminal.mapping.canonical_key
        self.state.geometry_hash = terminal.mapping.geometry_hash
        reward = energy_reward + novelty_reward
        self.state.history.append(HistoryEntry(text, "success", terminal.result.energy - old_energy,
                                               request.key, terminal.mapping.canonical_key, reward))
        return StepResult(True, "success", reward, energy_reward, novelty_reward, cache_hit,
                          request_key=request.key, terminal_key=terminal.mapping.canonical_key)


class NoveltyGroup:
    """Freeze one arm/seed history; publish the union only after all G rollouts."""
    def __init__(self, history: set[str], group_size=4):
        if type(group_size) is not int or group_size != 4:
            raise ValueError("Review08 requires exactly four rollouts per group")
        self.history = history
        self.frozen_history = frozenset(history)
        self.group_size = group_size
        self._visits = []
        self._states = []
        self._namespace = None
        self._merged = False

    def register(self, state: EpisodeState):
        if self._merged or len(self._visits) >= self.group_size or any(state is existing for existing in self._states):
            raise ValueError("Each group rollout may be registered exactly once")
        if state.frozen_history != self.frozen_history:
            raise ValueError("Rollout did not use this group's frozen history")
        if self._namespace is not None and state.namespace != self._namespace:
            raise ValueError("Group rollouts must belong to one arm/seed namespace")
        self._namespace = state.namespace
        self._states.append(state)
        self._visits.append(set(state.private_visits))

    def merge(self) -> frozenset[str]:
        if self._merged or len(self._visits) != self.group_size:
            raise ValueError("Merge requires one completed group and runs once")
        self.history.update(set().union(*self._visits))
        self._merged = True
        return frozenset(self.history)


def encode_occupancy(occupancy) -> dict[str, list[int]]:
    """Lossless compact occupation: all other parent cations/oxygens are Zr/O."""
    values = np.asarray(validate_occupancy(occupancy))
    return {"Y_sites": np.flatnonzero(values == 39).tolist(),
            "vacancy_sites": np.flatnonzero(values == 0).tolist()}


def decode_occupancy(payload) -> tuple[int, ...]:
    if not isinstance(payload, dict) or set(payload) != {"Y_sites", "vacancy_sites"}:
        raise ValueError("Compact occupation requires complete Y_sites and vacancy_sites lists")
    values = _parent().numbers.copy()
    for key, count, element, mother in [("Y_sites", 16, 39, 40), ("vacancy_sites", 8, 0, 8)]:
        sites = payload[key]
        if (not isinstance(sites, list) or len(sites) != count or any(type(i) is not int for i in sites)
                or len(set(sites)) != count or min(sites) < 0 or max(sites) >= 324):
            raise ValueError("Invalid complete compact site list: " + key)
        if not np.all(_parent().numbers[sites] == mother):
            raise ValueError("Compact site list crosses mother sublattices")
        values[sites] = element
    return validate_occupancy(values)


def encode_prompt_v2(state: EpisodeState, *, dense_occupancy=False) -> list[dict[str, str]]:
    """Return chat messages without tokenization or occupancy truncation.

    The caller applies the official tokenizer/chat template with thinking off,
    checks prompt <=1984 and total <=2048, and rejects oversized encodings.
    """
    # Derive the index rule from reference(3), and assert that the compact block
    # description exactly recovers every mother coordinate before presenting it.
    basis = np.rint(_parent().positions[:12] / 1.285).astype(int)
    blocks = np.arange(324) // 12
    xyz = np.column_stack((blocks // 9, (blocks // 3) % 3, blocks % 3))
    reconstructed = 1.285 * (4 * xyz + basis[np.arange(324) % 12])
    if not np.allclose(reconstructed, _parent().positions, atol=1e-12, rtol=0):
        raise ValueError("Mother site order changed; compact indexing cannot be used")
    remaining_wall_seconds = None
    if state.limits.wall_seconds is not None:
        elapsed = state.clock() - state.started_at
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Remaining wall time requires a finite monotonic elapsed time")
        remaining_wall_seconds = max(0.0, state.limits.wall_seconds - elapsed)
    occupation = list(validate_occupancy(state.occupancy)) if dense_occupancy else encode_occupancy(state.occupancy)
    observation = {"protocol": PROTOCOL, "occupancy": occupation,
                   "parent": {"cell_A": 15.42, "quarter_a_A": 1.285, "basis_quarter_a": basis.tolist()},
                   "energy_eV": state.energy, "best_energy_eV": state.best_energy,
                   "remaining_attempts": max(0, state.limits.max_attempts - state.steps_taken),
                   "remaining_wall_seconds": remaining_wall_seconds,
                   "budgets": asdict(state.limits), "counts": asdict(state.counters),
                   "recent": [{"action": item.action, "status": item.status, "dE": item.energy_delta}
                              for item in state.history[-3:]]}
    system = ("Return exactly one JSON object with keys op and sites, no other text. "
              "sites is two distinct integer mother-site indices 0..323. Occupancy codes: "
              "Zr=40,Y=39,O=8,vacancy=0. swap_cations swaps Y/Zr on the cation mother sublattice, "
              "parent minimum-image distance <=5.14 Angstrom. move_vacancy swaps vacancy/O on "
              "the oxygen mother sublattice, distance <=3.63452886 Angstrom. Tolerance 1e-6 Angstrom. "
              "Compact occupancy lists every Y and vacancy; all other parent cations are Zr and anions O. "
              "For index i, b=i//12,r=i%12; its coordinate in quarter-a units is "
              "4*(b//9,(b//3)%3,b%3)+basis_quarter_a[r], periodic with period 12. "
              "r<4 is cation, r>=4 is anion. "
              "Use the current terminal occupancy. Lower energy and new terminals earn reward.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(observation, separators=(",", ":"), allow_nan=False)}]
