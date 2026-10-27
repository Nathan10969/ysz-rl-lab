"""One preregistered, bounded local CPU diagnostic of saved teacher652.

This independent diagnostic does not modify any production calculator, runner,
protocol, saved652 artifact or Slurm ledger. No CUDA/Slurm/network operations.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import shutil
import signal
import time
import traceback

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import find_mic
from ase.io import write
from ase.io.jsonio import default as ase_json_default
from ase.optimize import BFGS, LBFGS, FIRE

import claude08_env as env
from smoke_common import MODEL_SHA, save_json, sha256, versions
from smoke_mace import build_batch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "training_smoke/claude10/recovery_v21/cpu_diagnostic"
INPUT_PATH = ROOT / "training_smoke/claude10/teacher/outputs/teacher_652/raw_candidates.jsonl"
SEED = 202610240652
WALL_SECONDS = 1800
RAW_EF_CAP = 2600


def append_json(path, row):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def timed_call(totals, key, function, *args, clock=time.monotonic, **kwargs):
    begin = clock()
    try:
        return function(*args, **kwargs)
    finally:
        totals[key] = totals.get(key, 0.) + clock() - begin


def geometry_key(atoms):
    """Exact bytes; no rounded positions or tolerance-based graph reuse."""
    digest = hashlib.sha256()
    for name, value in [("cell", atoms.cell.array), ("pbc", atoms.pbc),
                        *sorted(atoms.arrays.items())]:
        array = np.asarray(value)
        digest.update(name.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps(atoms.info, sort_keys=True, allow_nan=False,
                            default=ase_json_default).encode())
    return digest.hexdigest()


def directional_inputs(atoms, parent, *, atom_index=255, source=261, target=257, distance=.1):
    if atoms.numbers[atom_index] != 8 or parent.numbers[source] != 8 or parent.numbers[target] != 8:
        raise ValueError("Oxygen atom and mother oxygen sites required")
    delta = parent.positions[target] - parent.positions[source]
    vector = find_mic(delta, parent.cell, pbc=parent.pbc)[0]
    direction = vector / np.linalg.norm(vector)
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions[atom_index] += distance * direction
    minus.positions[atom_index] -= distance * direction
    return plus, minus, direction


def seeded_displacement(atoms, *, seed=SEED, sigma=.02):
    result = atoms.copy()
    result.positions += np.random.default_rng(seed).normal(0., sigma, result.positions.shape)
    return result


def matrix_definitions():
    cases = [{"name": "A_O255_plus", "record_id": 31, "optimizer": "BFGS", "displacement_A": .1},
             {"name": "A_O255_minus", "record_id": 31, "optimizer": "BFGS", "displacement_A": -.1}]
    cases += [{"name": f"B_record{rid}_{optimizer}", "record_id": rid, "optimizer": optimizer,
               "seed": SEED, "sigma_A": .02, "subtract_mean": False}
              for rid in (31, 30) for optimizer in ("BFGS", "LBFGS", "FIRE")]
    return [dict(case, fmax=.03, steps=300, maxstep=.1, dtype="float64", device="cpu") for case in cases]


class ExactGraphCache:
    """Candidate only: one immutable CPU graph, exact state match, cloned returns."""
    def __init__(self, builder):
        self.builder = builder
        self.key = self.batch = None
        self.hits = self.misses = 0

    def get(self, atoms):
        key = geometry_key(atoms)
        if key != self.key:
            self.batch = self.builder(atoms)
            self.key = key
            self.misses += 1
        else:
            self.hits += 1
        return self.batch.clone()


class EFProbe:
    """Charge actual singleton graphs in model pre-hook, including failed calls."""
    def __init__(self, model, metadata, output, *, counter):
        self.model, self.metadata, self.output = model, metadata, Path(output)
        self.counter = counter
        self.case = None
        self.context = None
        self.attempts = []
        self.latest = {}
        self.timings = {}
        self.pre_hook = model.register_forward_pre_hook(self._before, with_kwargs=True)
        self.post_hook = model.register_forward_hook(self._after, with_kwargs=True, always_call=True)

    def begin_case(self, name):
        self.case = name
        self.timings.setdefault(name, {"graph_seconds": 0., "forward_seconds": 0.,
            "calculator_seconds": 0., "optimizer_step_inclusive_seconds": 0.,
            "optimizer_step_calls": 0, "optimizer_step_graph_overlap_seconds": 0.,
            "optimizer_step_forward_overlap_seconds": 0., "mapping_seconds": 0.})

    def check(self):
        self.counter.check_time()

    def graph(self, atoms):
        self.check()
        return timed_call(self.timings[self.case], "graph_seconds", build_batch,
                          self.metadata, [atoms], "cpu")

    def _before(self, model, args, kwargs):
        if self.context is None or self.context.get("charged"):
            raise RuntimeError("Explicit singleton diagnostic context required")
        payload = args[0] if args else kwargs.get("data")
        size = int(payload["ptr"].numel() - 1)
        if size != 1:
            raise RuntimeError("Diagnostic permits only actual singleton graph payloads")
        self.counter.before_forward(size)
        counts = self.counter.counters
        event = {"case": self.case, "forward_call": counts.forward_calls,
            "raw_ef_cumulative": counts.raw_ef, "actual_graphs": size, "status": "started",
            "geometry_sha256": geometry_key(self.context["atoms"]),
            "elapsed_seconds": self.counter.clock() - self.counter.started_at}
        self.context.update(charged=True, event=event)
        self.attempts.append(event)
        append_json(self.output / "ef_attempts.jsonl", event)
        self.context["forward_start"] = time.monotonic()

    def _after(self, model, args, kwargs, output):
        if self.context is None or not self.context.get("charged"):
            return
        event = self.context["event"]
        seconds = time.monotonic() - self.context["forward_start"]
        self.timings[self.case]["forward_seconds"] += seconds
        event["forward_seconds"] = seconds
        try:
            if output is None:
                raise RuntimeError("Model forward failed before returning E/F")
            energy = output["energy"].detach().cpu().numpy().reshape(-1) * self.metadata.energy_units_to_eV
            forces = output["forces"].detach().cpu().numpy().copy()
            forces *= self.metadata.energy_units_to_eV / self.metadata.length_units_to_A
            atoms = self.context["atoms"]
            if (energy.shape != (1,) or forces.shape != (len(atoms), 3)
                    or not np.isfinite(energy).all() or not np.isfinite(forces).all()):
                raise RuntimeError("Nonfinite or malformed diagnostic E/F")
            event.update(status="completed", energy_eV=float(energy[0]),
                         fmax_eV_A=float(np.linalg.norm(forces, axis=1).max()), forces_eV_A=forces.tolist())
            self.latest[self.case] = {"atoms": atoms.copy(), "energy_eV": float(energy[0]),
                                      "forces": forces.copy(), "geometry_sha256": event["geometry_sha256"]}
            self.context["result"] = (float(energy[0]), forces)
        except Exception as exc:
            event.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
            if output is not None:
                raise
        finally:
            append_json(self.output / "ef_calls.jsonl", event)

    def evaluate(self, atoms, payload):
        if self.context is not None:
            raise RuntimeError("Nested diagnostic model call")
        self.context = {"atoms": atoms.copy(), "charged": False}
        try:
            self.model(payload, compute_stress=False, training=False,
                       compute_edge_forces=False, compute_atomic_stresses=False)
            return self.context["result"]
        finally:
            self.context = None

    def close(self):
        self.pre_hook.remove()
        self.post_hook.remove()


class DiagnosticCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, probe):
        super().__init__()
        self.probe = probe

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        begin = time.monotonic()
        try:
            super().calculate(atoms, properties, system_changes)
            batch = self.probe.graph(self.atoms)
            energy, forces = self.probe.evaluate(self.atoms, batch.to_dict())
            self.results = {"energy": energy, "free_energy": energy, "forces": forces}
        finally:
            self.probe.timings[self.probe.case]["calculator_seconds"] += time.monotonic() - begin


def mapping_of(atoms):
    try:
        mapping = env.map_terminal(atoms)
        return {"status": "mapped", **asdict(mapping)}
    except env.MappingError as exc:
        return {"status": "unmappable", "error": str(exc)}


def write_evaluated(path, latest):
    atoms = latest["atoms"].copy()
    atoms.calc = SinglePointCalculator(atoms, energy=latest["energy_eV"], forces=latest["forces"])
    write(path, atoms, format="extxyz")


def run_case(definition, initial, probe, output, saved31, parent, direction, heartbeat):
    name = definition["name"]
    folder = output / name
    folder.mkdir(exist_ok=False)
    write(folder / "initial.extxyz", initial, format="extxyz")
    probe.begin_case(name)
    before = probe.counter.counters.raw_ef
    start = time.monotonic()
    result = dict(definition, status="started", initial_geometry_sha256=geometry_key(initial))
    heartbeat(name, result)
    atoms = initial.copy()
    atoms.calc = DiagnosticCalculator(probe)
    optimizer = None
    try:
        probe.check()
        cls = {"BFGS": BFGS, "LBFGS": LBFGS, "FIRE": FIRE}[definition["optimizer"]]
        optimizer = cls(atoms, maxstep=.1, logfile=str(folder / "optimizer.log"),
                        trajectory=str(folder / "optimization.traj"))
        result["optimizer_parameters"] = {"constructor_defaults": str(inspect.signature(cls)),
                                            "explicit": {"maxstep": .1}, "line_search": False}
        step = optimizer.step
        def measured_step(*args, **kwargs):
            probe.check()
            timings = probe.timings[name]
            graph_before, forward_before = timings["graph_seconds"], timings["forward_seconds"]
            try:
                return timed_call(timings, "optimizer_step_inclusive_seconds", step, *args, **kwargs)
            finally:
                timings["optimizer_step_calls"] += 1
                timings["optimizer_step_graph_overlap_seconds"] += timings["graph_seconds"] - graph_before
                timings["optimizer_step_forward_overlap_seconds"] += timings["forward_seconds"] - forward_before
        optimizer.step = measured_step
        def ionic_progress():
            probe.check()
            if optimizer.nsteps % 25 == 0:
                heartbeat(name, dict(result, nsteps=optimizer.nsteps,
                    raw_ef=probe.counter.counters.raw_ef - before,
                    wall_seconds=time.monotonic() - start))
        optimizer.attach(ionic_progress)
        converged = bool(optimizer.run(fmax=.03, steps=300))
        result.update(status="completed" if converged else "unconverged", converged=converged)
    except env.BudgetExceeded as exc:
        result.update(status="truncated", converged=False, error=str(exc))
    except Exception as exc:
        result.update(status="failed", converged=False, error=type(exc).__name__ + ": " + str(exc),
                      traceback=traceback.format_exc())
    finally:
        result.update(nsteps=None if optimizer is None else optimizer.nsteps,
                      raw_ef=probe.counter.counters.raw_ef - before)
        latest = probe.latest.get(name)
        if latest is not None:
            write_evaluated(folder / "final_evaluated.extxyz", latest)
            fmax = float(np.linalg.norm(latest["forces"], axis=1).max())
            result.update(energy_eV=latest["energy_eV"], fmax_eV_A=fmax,
                          final_evaluated_geometry_sha256=latest["geometry_sha256"],
                          force_converged=fmax <= .03)
            if result["status"] == "truncated":
                result["mapping"] = {"status": "not_measured_after_budget_stop"}
            else:
                result["mapping"] = timed_call(probe.timings[name], "mapping_seconds",
                                                mapping_of, latest["atoms"])
            result["strict_terminal_valid"] = bool(result.get("converged") and result["mapping"]["status"] == "mapped")
            if name.startswith("A_"):
                delta = find_mic(latest["atoms"].positions[255] - saved31.positions[255],
                                 saved31.cell, pbc=True)[0]
                distances = [float(find_mic(latest["atoms"].positions[255] - parent.positions[site],
                                           parent.cell, pbc=True)[1]) for site in (261, 257)]
                result["directional_observation"] = {"O255_net_projection_from_saved31_A": float(delta @ direction),
                    "O255_distance_to_mother_261_257_A": distances,
                    "O255_final_position_A": latest["atoms"].positions[255].tolist(),
                    "O255_force_norm_eV_A": float(np.linalg.norm(latest["forces"][255]))}
        write(folder / "current_positions.extxyz", atoms.copy(), format="extxyz")
        result.update(wall_seconds=time.monotonic() - start, timings=dict(probe.timings[name]))
        save_json(folder / "result.json", result)
        heartbeat(name, result)
    return result


def graph_candidate(atoms, metadata, probe, output):
    """Five graph builds/hits each; exactly one E/F comparison pair, no relaxation."""
    probe.begin_case("C_exact_geometry_cache")
    folder = output / "C_exact_geometry_cache"
    folder.mkdir(exist_ok=False)
    write(folder / "saved_record31.extxyz", atoms, format="extxyz")
    baseline_times, cache_times = [], []
    native = None
    for _ in range(5):
        probe.check()
        begin = time.monotonic()
        native = build_batch(metadata, [atoms], "cpu")
        baseline_times.append(time.monotonic() - begin)
    cache = ExactGraphCache(lambda cell: build_batch(metadata, [cell], "cpu"))
    begin = time.monotonic()
    cache.get(atoms)
    miss_seconds = time.monotonic() - begin
    candidate = None
    for _ in range(5):
        probe.check()
        begin = time.monotonic()
        candidate = cache.get(atoms)
        cache_times.append(time.monotonic() - begin)
    native_e, native_f = probe.evaluate(atoms, native.to_dict())
    candidate_e, candidate_f = probe.evaluate(atoms, candidate.to_dict())
    de, df = abs(native_e - candidate_e), float(np.abs(native_f - candidate_f).max())
    result = {"status": "completed" if de <= 1e-6 and df <= 1e-6 else "parity_failed",
        "candidate": "exact identical complete ASE state graph reuse; immutable cached batch with cloned return",
        "native_build_batch_seconds": baseline_times, "cache_miss_seconds": miss_seconds,
        "cache_hit_seconds_including_exact_key_and_clone": cache_times,
        "cache_hits": cache.hits, "cache_misses": cache.misses,
        "energy_difference_eV": de, "force_difference_eV_A": df, "tolerance": 1e-6,
        "native_energy_eV": native_e, "candidate_energy_eV": candidate_e,
        "native_graph_median_seconds": float(np.median(baseline_times)),
        "cache_hit_median_seconds": float(np.median(cache_times)),
        "scope_limit": "Only identical-state hits measured. Ionic position changes invalidate the cache; no changing-geometry or ASE-skin speedup claim.",
        "GPU_neighbor_builder": "not measured; no GPU execution authorized", "raw_ef": 2}
    write_evaluated(folder / "candidate_single_point.extxyz", probe.latest[probe.case])
    save_json(folder / "result.json", result)
    if result["status"] != "completed":
        raise RuntimeError("Exact graph candidate failed real E/F parity")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/mace-small-density-agnesi-stress.model")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    counter = env.RawEFCounter(env.Counters(), env.BudgetLimits(max_raw_ef=RAW_EF_CAP,
                                wall_seconds=WALL_SECONDS), started_at=start)
    result = {"status": "initializing", "device": "cpu", "GPU_work_performed": False,
        "Slurm_work_performed": False, "CPU_pool_only": True,
        "limits": {"global_wall_seconds": WALL_SECONDS, "total_structure_EF": RAW_EF_CAP},
        "cases": {row["name"]: dict(row, status="planned") for row in matrix_definitions()},
        "interpretation_limit": "Two O255 perturbations test directional endpoint stability only; no complete saddle/minimum or Hessian classification.",
        "timing_scope": {"graph_seconds": "Complete smoke_mace.build_batch including CPU graph construction and Batch transfer; no production source patch.",
            "forward_seconds": "Actual model pre/post-hook interval, compute_stress=False; excludes graph and durable-attempt serialization.",
            "calculator_seconds": "Whole diagnostic ASE Calculator.calculate including graph, model, postprocessing and logs; overlaps graph/forward.",
            "optimizer_step_inclusive_seconds": "Wrapped optimizer instance.step, including cache reads and array updates plus any nested E/F. Not exclusive Hessian time.",
            "optimizer_step_overlaps": "Graph and forward blocks executed inside step are separately counted; do not add inclusive overlapping totals.",
            "mapping_seconds": "Unchanged env.map_terminal after evaluated final geometry; excluded from relaxation steps."}}
    probe = None
    def heartbeat(name, row):
        result["cases"][name] = row
        result.update(counters=asdict(counter.counters), wall_seconds=time.monotonic() - start)
        save_json(output / "progress.json", result)
        append_json(output / "case_events.jsonl", {"name": name, "status": row["status"],
            "nsteps": row.get("nsteps"), "raw_ef": row.get("raw_ef"),
            "elapsed_seconds": result["wall_seconds"]})
        print(json.dumps({"case": name, "status": row["status"], "steps": row.get("nsteps"),
            "raw_ef": row.get("raw_ef"), "wall_seconds": result["wall_seconds"]}), flush=True)
    def alarm(signum, frame):
        raise env.BudgetExceeded("Global 1800-second CPU diagnostic safety deadline reached")
    previous_alarm = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS)
    try:
        rows = [json.loads(line) for line in INPUT_PATH.read_text().splitlines()]
        records = {row["id"]: row for row in rows}
        saved = records[31]["result"]
        saved31 = Atoms(numbers=saved["numbers"], positions=saved["positions"], cell=saved["cell"], pbc=saved["pbc"])
        parent = env._parent()
        plus, minus, direction = directional_inputs(saved31, parent)
        perturbed = {rid: seeded_displacement(env.build_request(records[rid]["request_occupancy"]).atoms)
                     for rid in (31, 30)}
        inputs = {"A_O255_plus": plus, "A_O255_minus": minus}
        inputs.update({row["name"]: perturbed[row["record_id"]].copy() for row in matrix_definitions()[2:]})
        plan = {"matrix": [dict(row, initial_geometry_sha256=geometry_key(inputs[row["name"]]))
                           for row in matrix_definitions()],
                "fixed_seed": SEED, "Gaussian_mean_removed": False, "atom_and_site_indexing": "zero based",
                "O255_direction_source_to_target": [261, 257], "minimum_image_unit_direction": direction.tolist(),
                "saved31_energy_eV": saved["energy_eV"], "input_journal_sha256": sha256(INPUT_PATH),
                "checkpoint_sha256": sha256(args.checkpoint), "retries": 0,
                "C_graph_repetitions_each": 5, "C_real_EF_comparison_pairs": 1}
        if plan["checkpoint_sha256"] != MODEL_SHA:
            raise ValueError("Frozen checkpoint SHA256 mismatch")
        save_json(output / "plan.json", plan)
        sources = output / "sources"; sources.mkdir()
        for path in (Path(__file__), ROOT / "tests/test_review10_652_cpu.py"):
            shutil.copyfile(path, sources / path.name)
        result["source_sha256"] = {name: sha256(ROOT / "scripts" / name) for name in
                                      ("review10_652_cpu.py", "claude08_env.py", "smoke_mace.py")}
        result["tests_sha256"] = sha256(ROOT / "tests/test_review10_652_cpu.py")
        result["input_journal_sha256"] = plan["input_journal_sha256"]
        result["checkpoint_sha256"] = MODEL_SHA
        result["saved31_energy_eV"] = saved["energy_eV"]
        save_json(output / "progress.json", result)
        import torch
        from mace.calculators import MACECalculator
        torch.set_num_threads(2)
        torch.set_default_dtype(torch.float64)
        load_begin = time.monotonic()
        metadata = MACECalculator(model_paths=str(args.checkpoint), device="cpu", default_dtype="float64",
                                  enable_cueq=False, enable_oeq=False)
        if len(metadata.models) != 1:
            raise ValueError("Exactly one frozen MACE model required")
        model = metadata.models[0]
        model.eval()
        for parameter in model.parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float64:
                raise ValueError("Float64 MACE parameters required")
            parameter.requires_grad_(False)
        result.update(load_seconds=time.monotonic() - load_begin, torch_threads=torch.get_num_threads(),
                      versions=versions(["torch", "mace-torch", "ase", "numpy", "scipy"]),
                      optimizer_source_sha256={cls.__name__: sha256(inspect.getsourcefile(cls))
                                               for cls in (BFGS, LBFGS, FIRE)})
        probe = EFProbe(model, metadata, output, counter=counter)
        result["status"] = "running"
        heartbeat("C_exact_geometry_cache", {"status": "started"})
        result["graph_candidate"] = graph_candidate(saved31, metadata, probe, output)
        heartbeat("C_exact_geometry_cache", result["graph_candidate"])
        budget_stop = False
        for definition in matrix_definitions():
            if budget_stop:
                heartbeat(definition["name"], dict(definition, status="not_started", reason="global budget stop; no retry"))
                continue
            measurement = run_case(definition, inputs[definition["name"]], probe, output,
                                   saved31, parent, direction, heartbeat)
            budget_stop = measurement["status"] == "truncated"
        result["status"] = "truncated" if budget_stop else "completed"
    except Exception as exc:
        result.update(status="truncated" if isinstance(exc, env.BudgetExceeded) else "failed",
                      error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        if probe is not None:
            probe.close()
            result["timings"] = probe.timings
            result["actual_model_pre_hook_calls"] = len(probe.attempts)
            result["actual_graph_sum"] = sum(row["actual_graphs"] for row in probe.attempts)
            result["failed_model_forwards"] = sum(row["status"] == "failed" for row in probe.attempts)
        result.update(counters=asdict(counter.counters), wall_seconds=time.monotonic() - start)
        result["all_registered_relaxations_attempted"] = all(row["status"] not in ("planned", "not_started")
             for name, row in result["cases"].items() if name != "C_exact_geometry_cache")
        save_json(output / "progress.json", result)
        save_json(output / "result.json", result)
    print(json.dumps({"status": result["status"], "raw_ef": counter.counters.raw_ef,
                      "forward_calls": counter.counters.forward_calls,
                      "wall_seconds": result["wall_seconds"], "output": str(output)}), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
