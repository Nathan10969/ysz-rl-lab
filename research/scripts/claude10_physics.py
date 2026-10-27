"""Review10 frozen-MACE E/F batches and independent fixed-cell ASE BFGS.

Only explicit ``device='cpu'`` permits local model execution. CUDA model loading
uses the production single-Slurm-GPU guard. Importing this module loads no model.
Raw E/F means one evaluated structure, so a forward of B graphs costs B.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import time

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import BFGS

import claude08_env as env
from smoke_common import sha256, save_json, versions
from smoke_mace import build_batch


@dataclass
class BatchPhysicsResult(env.PhysicsResult):
    evidence: dict = field(default_factory=dict)


class BatchPhysicsError(RuntimeError):
    """A batch failed; evidence retains attempts, but supplies no invented E/F."""
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def _load_calculator(checkpoint, device):
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=str(checkpoint), device=device,
                          default_dtype="float64", enable_cueq=False, enable_oeq=False)


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name: sha256(root / name) for name in
            ("claude10_physics.py", "claude08_env.py", "smoke_mace.py", "smoke_grpo08.py")}


class ForwardAudit:
    """Read-only model pre-hook independently checks charges against real graphs."""
    def __init__(self, counter, log=None):
        self.counter, self.log = counter, log
        self.raw_start = self.last_raw = counter.counters.raw_ef
        self.last_call = counter.counters.forward_calls
        self.rows = []
        self.observed_raw_ef = 0

    def __call__(self, model, args, kwargs):
        payload = args[0] if args else kwargs["data"]
        size = int(payload["ptr"].numel() - 1)
        counts = self.counter.counters
        if (not 1 <= size <= 4 or counts.raw_ef - self.last_raw != size
                or counts.forward_calls - self.last_call != 1):
            raise RuntimeError("Actual model graphs must be precharged exactly once")
        row = {"batch_size": size, "raw_ef_cumulative": counts.raw_ef,
               "forward_call": counts.forward_calls, "precharged": True}
        self.rows.append(row)
        self.observed_raw_ef += size
        self.last_raw, self.last_call = counts.raw_ef, counts.forward_calls
        if self.log is not None:
            self.log(row)


class BatchMACEPhysics:
    """One frozen model; 1--4 true graphs and separate ASE Hessians per request.

    ``counter`` is one shared env.RawEFCounter. Callers own ``relax_actual``;
    this backend charges only actual raw E/F forwards. Results keep input order.
    Optional actors implement ``to_device('cpu', phaseledger)``. When supplied,
    ``counter.phase_ledger`` is preferred; a physics-only ledger is the fallback.
    """
    def __init__(self, checkpoint, output, device="cuda", actor=None):
        if device not in ("cpu", "cuda"):
            raise ValueError("Explicit cpu or cuda device required")
        self.checkpoint, self.output = Path(checkpoint), Path(output)
        self.device, self.actor = device, actor
        self.checkpoint_sha = sha256(self.checkpoint)
        if self.checkpoint_sha != env.MODEL_SHA:
            raise ValueError("Frozen MACE checkpoint SHA256 mismatch")
        self.source_hashes = source_hashes()
        self.calc = None
        self.torch = None
        self.operation = 0
        self.output.mkdir(parents=True, exist_ok=True)
        self.last_evidence = []
        self._forward_hook = None
        self._forward_context = None

    def _append(self, filename, row):
        with (self.output / filename).open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    def _sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def _ensure(self, counter):
        counter.check_time()
        if self.actor is not None:
            ledger = getattr(counter, "phase_ledger", None)
            if ledger is None:
                ledger = SimpleNamespace(check=counter.check_time, counters=counter.counters,
                    limits=counter.limits, clock=counter.clock, started_at=counter.started_at,
                    phase_timings=[], log_path=None)
            self.actor.to_device("cpu", ledger)
        if self.device == "cuda":
            from smoke_grpo08 import require_single_gpu_job
            require_single_gpu_job()
        if self.torch is None:
            import torch
            self.torch = torch
        if self.calc is None:
            if sha256(self.checkpoint) != self.checkpoint_sha:
                raise ValueError("Frozen MACE checkpoint SHA256 changed before load")
            # AtomicData uses the global default for graph floating tensors.
            self.torch.set_default_dtype(self.torch.float64)
            self.calc = _load_calculator(self.checkpoint, self.device)
        else:
            self.calc.models[0].to(self.device)
        if len(self.calc.models) != 1:
            raise ValueError("Exactly one frozen MACE model required")
        model = self.calc.models[0]
        model.eval()
        for parameter in model.parameters():
            if parameter.is_floating_point() and parameter.dtype != self.torch.float64:
                raise ValueError("MACE parameters must all be float64")
            parameter.requires_grad_(False)
        for buffer in model.buffers():
            if buffer.is_floating_point() and buffer.dtype != self.torch.float64:
                raise ValueError("MACE buffers must all be float64")
        if self._forward_hook is None:
            self._forward_hook = model.register_forward_pre_hook(self._before_forward, with_kwargs=True)
        counter.check_time()

    @contextmanager
    def _forward_scope(self, counter, event, *, expected_size, active=(), receipts=(),
                       attempt_log="ef_attempts.jsonl"):
        if self._forward_context is not None:
            raise RuntimeError("Nested physical forward context is forbidden")
        self._forward_context = SimpleNamespace(counter=counter, event=event,
            expected_size=expected_size, active=active, receipts=receipts,
            attempt_log=attempt_log, charged=False)
        try:
            yield
        finally:
            self._forward_context = None

    def _before_forward(self, model, args, kwargs):
        """Guard every real model invocation and charge its actual graph count."""
        context = self._forward_context
        if context is None or context.charged:
            raise RuntimeError("One explicit counter context is required per physical forward")
        payload = args[0] if args else kwargs.get("data")
        ptr = payload.get("ptr") if isinstance(payload, dict) else None
        if ptr is None or ptr.ndim != 1:
            raise RuntimeError("Actual graph payload must identify its request batch")
        size = int(ptr.numel() - 1)
        context.event.update(actual_graphs=size, intended_graphs=context.expected_size)
        if not 1 <= size <= 4 or size != context.expected_size:
            raise RuntimeError("Actual graph count does not match the request batch")
        context.counter.before_forward(size)
        context.charged = True
        counts = context.counter.counters
        context.event.update(batch_size=size, forward_call=counts.forward_calls,
                             raw_ef_cumulative=counts.raw_ef)
        for lane in context.active:
            context.receipts[lane]["raw_ef_attempts"] += 1
            context.receipts[lane]["forward_calls"].append(counts.forward_calls)
        self._append(context.attempt_log, context.event)

    def _offload(self):
        if self.calc is not None and self.device == "cuda":
            self._sync()
            self.calc.models[0].to("cpu")
            self.torch.cuda.empty_cache()

    def _validate(self, requests, max_steps):
        requests = list(requests)
        if not 1 <= len(requests) <= 4:
            raise ValueError("A physical batch must contain 1--4 requests")
        if type(max_steps) is not int or max_steps not in ((1, 300) if self.device == "cpu" else (300,)):
            raise ValueError("Formal BFGS requires 300 steps; CPU preflight may tighten to 1")
        for request in requests:
            if not isinstance(request, env.PhysicsRequest):
                raise ValueError("PhysicsRequest required")
            if request.spec != env.RelaxationSpec() or request.spec.dtype != "float64":
                raise ValueError("Request must use frozen SHA and float64 BFGS/.03/300/maxstep .1")
            atoms = request.atoms
            if (not len(atoms) or not np.isfinite(atoms.positions).all()
                    or not np.isfinite(atoms.cell.array).all() or atoms.constraints):
                raise ValueError("Finite unconstrained fixed-cell atoms required")
        return requests

    def _evaluate(self, atoms, active, receipts, counter, phase):
        counter.check_time()
        self._sync()
        begin = time.monotonic()
        batch = build_batch(self.calc, [atoms[i] for i in active], self.device)
        payload = batch.to_dict()
        # Match MACECalculator's model-dtype conversion without de-batching.
        for name, value in payload.items():
            if self.torch.is_tensor(value) and value.is_floating_point():
                payload[name] = value.to(dtype=self.torch.float64)
        self._sync()
        graph_seconds = time.monotonic() - begin
        event = {"operation": self.operation, "phase": phase, "lanes": list(active),
                 "request_keys": [receipts[i]["request_key"] for i in active],
                 "graph_seconds": graph_seconds}
        forward_start = time.monotonic()
        try:
            with self._forward_scope(counter, event, expected_size=len(active), active=active, receipts=receipts):
                output = self.calc.models[0](payload, compute_stress=False, training=False,
                             compute_edge_forces=False, compute_atomic_stresses=False)
            # ``energy`` includes the chosen head's atomic reference and scale.
            energy = output["energy"].detach().cpu().numpy().copy()
            forces = output["forces"].detach().cpu().numpy().copy()
            energy = np.asarray(energy).reshape(-1) * self.calc.energy_units_to_eV
            forces *= self.calc.energy_units_to_eV / self.calc.length_units_to_A
            self._sync()
            if (energy.shape != (len(active),) or forces.shape != (sum(len(atoms[i]) for i in active), 3)
                    or not np.isfinite(energy).all() or not np.isfinite(forces).all()):
                raise RuntimeError("Malformed or nonfinite MACE E/F output")
        except Exception as exc:
            event.update(status="failed", error=type(exc).__name__ + ": " + str(exc),
                         forward_seconds=time.monotonic() - forward_start,
                         seconds=time.monotonic() - begin)
            self._append("ef_calls.jsonl", event)
            for i in active:
                receipts[i]["forward_timings"].append(dict(event))
            raise
        split = np.split(forces, np.cumsum([len(atoms[i]) for i in active])[:-1])
        event.update(status="completed", forward_seconds=time.monotonic() - forward_start,
                     seconds=time.monotonic() - begin, energy_eV=energy.tolist(),
                     fmax_eV_A=[float(np.linalg.norm(f, axis=1).max()) for f in split])
        self._append("ef_calls.jsonl", event)
        for i in active:
            receipts[i]["forward_timings"].append(dict(event))
        return energy, split

    def _run(self, requests, counter, *, max_steps, single_point):
        requests = self._validate(requests, max_steps)
        self.operation += 1
        begin = time.monotonic()
        atoms = [request.atoms.copy() for request in requests]
        receipts = [{"operation": self.operation, "lane": i, "request_key": request.key,
                     "checkpoint_sha256": self.checkpoint_sha, "source_hashes": self.source_hashes,
                     "device": self.device, "dtype": "float64", "optimizer": "BFGS",
                     "fmax_target_eV_A": .03, "maxstep_A": .1, "protocol_max_steps": 300,
                     "execution_max_steps": 0 if single_point else max_steps,
                     "raw_ef_attempts": 0, "nsteps": 0, "forward_calls": [],
                     "forward_timings": [], "status": "pending", "formal_receipt": False}
                    for i, request in enumerate(requests)]
        self.last_evidence = receipts
        last = {}
        try:
            self._ensure(counter)
            optimizers = [] if single_point else [BFGS(cell, logfile=None, maxstep=.1) for cell in atoms]
            active = list(range(len(atoms)))
            while active:
                energy, forces = self._evaluate(atoms, active, receipts, counter,
                                                "single_point" if single_point else "BFGS")
                next_active = []
                for local, lane in enumerate(active):
                    f = forces[local]
                    e = float(energy[local])
                    converged = bool(np.linalg.norm(f, axis=1).max() <= .03)
                    last[lane] = (e, f.copy(), converged)
                    if not single_point and not converged and optimizers[lane].nsteps < max_steps:
                        # ASE reads this exact E/F cache; no implicit MACE callback.
                        atoms[lane].calc = SinglePointCalculator(atoms[lane], energy=e, forces=f)
                        optimizers[lane].step()
                        optimizers[lane].nsteps += 1
                        receipts[lane]["nsteps"] = optimizers[lane].nsteps
                        next_active.append(lane)
                    else:
                        receipts[lane].update(status="single_point" if single_point else
                            ("converged" if converged else "step_capped_unconverged"),
                            formal_receipt=bool(not single_point and max_steps == 300 and converged),
                            energy_eV=e, fmax_eV_A=float(np.linalg.norm(f, axis=1).max()),
                            terminal_seconds=time.monotonic() - begin)
                active = next_active
            counter.check_time()
        except Exception as exc:
            for receipt in receipts:
                if receipt["status"] == "pending":
                    receipt.update(status="budget_exhausted" if isinstance(exc, env.BudgetExceeded) else "failed",
                                   error=type(exc).__name__ + ": " + str(exc))
            if isinstance(exc, env.BudgetExceeded):
                raise
            raise BatchPhysicsError(str(exc), receipts) from exc
        finally:
            self._offload()
            elapsed = time.monotonic() - begin
            for receipt in receipts:
                receipt["operation_seconds"] = elapsed
                self._append("physics_receipts.jsonl", receipt)
        return [BatchPhysicsResult(atoms[i].copy(), *last[i], evidence=receipts[i]) for i in range(len(atoms))]

    def relax_many(self, requests, counter, max_steps=300):
        return self._run(requests, counter, max_steps=max_steps, single_point=False)

    def single_points(self, requests, counter):
        return self._run(requests, counter, max_steps=300, single_point=True)

    def __call__(self, request, counter):
        return self.relax_many([request], counter)[0]

    def close(self):
        if self._forward_hook is not None:
            self._forward_hook.remove()
            self._forward_hook = None
        self._offload()
        self.calc = None


def cpu_execution_preflight(checkpoint, output):
    """Real batch1/4 SP, singleton/native parity, and one BFGS step per copy."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    import torch
    from ase.io import read, write
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    initial = read(root / "training_smoke/data/mace_initial_lattice.extxyz", index=":4")
    requests = [env.build_request(tuple(map(int, cell.numbers))) for cell in initial]
    if len(requests) != 4:
        raise ValueError("Four physical preflight inputs required")
    counter = env.RawEFCounter(env.Counters(), env.BudgetLimits(max_raw_ef=64, wall_seconds=1800))
    engine = BatchMACEPhysics(checkpoint, output, device="cpu")
    proof = {"status": "running", "runtime": {"device": "cpu", "GPU_work_performed": False,
             "torch_threads": torch.get_num_threads()}, "source_hashes": source_hashes(),
             "checkpoint_sha256": engine.checkpoint_sha, "production_code_path": True,
             "versions": versions(["torch", "mace-torch", "ase", "numpy", "scipy"]),
             "input_sha256": sha256(root / "training_smoke/data/mace_initial_lattice.extxyz"),
             "parity_tolerance": 1e-6, "short_receipts_for_training": False}
    begin = time.monotonic()
    audit = None
    audit_hook = None
    try:
        engine._ensure(counter)
        audit = ForwardAudit(counter, lambda row: engine._append("actual_model_forwards.jsonl", row))
        audit_hook = engine.calc.models[0].register_forward_pre_hook(audit, with_kwargs=True)
        singleton = [engine.single_points([request], counter)[0] for request in requests]
        batch = engine.single_points(requests, counter)
        native = []
        # Native parity uses the same production pre-hook and deliberate context.
        for request in requests:
            cell = request.atoms.copy()
            engine.calc.reset()
            cell.calc = engine.calc
            event = {"operation": engine.operation, "phase": "native_singleton",
                     "request_keys": [request.key]}
            with engine._forward_scope(counter, event, expected_size=1,
                                       attempt_log="native_ef_attempts.jsonl"):
                native.append((float(cell.get_potential_energy()), cell.get_forces().copy()))
        parity = []
        for lane in range(4):
            parity.append({"lane": lane,
                "batch_single_energy_difference_eV": abs(batch[lane].energy - singleton[lane].energy),
                "batch_single_force_difference_eV_A": float(np.abs(batch[lane].forces - singleton[lane].forces).max()),
                "batch_native_energy_difference_eV": abs(batch[lane].energy - native[lane][0]),
                "batch_native_force_difference_eV_A": float(np.abs(batch[lane].forces - native[lane][1]).max())})
        proof["parity"] = parity
        if any(value > 1e-6 for row in parity for key, value in row.items() if key != "lane"):
            raise RuntimeError("Batch/singleton/native MACE parity failed")
        relaxation = {}
        for size in (1, 4):
            results = engine.relax_many(requests[:size], counter, max_steps=1)
            relaxation[str(size)] = [result.evidence for result in results]
            for lane, result in enumerate(results):
                result.atoms.calc = SinglePointCalculator(result.atoms, energy=result.energy, forces=result.forces)
                write(output / f"bfgs_one_step_b{size}_lane{lane}.extxyz", result.atoms)
            if any(r.evidence["nsteps"] != 1 for r in results):
                raise RuntimeError("Preflight inputs must exercise one actual BFGS step")
        proof.update(status="cpu_execution_preflight_passed", relaxations=relaxation,
                     singleton_energies_eV=[r.energy for r in singleton],
                     native_energies_eV=[r[0] for r in native],
                     native_energy_semantics={"output": "energy", "head": engine.calc.head,
                         "includes_atomic_reference_and_scale": True,
                         "energy_units_to_eV": engine.calc.energy_units_to_eV,
                         "length_units_to_A": engine.calc.length_units_to_A})
    except Exception as exc:
        proof.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        if audit_hook is not None:
            audit_hook.remove()
        if audit is not None:
            proof["actual_forward_audit"] = {"observed_raw_ef": audit.observed_raw_ef,
                "observed_forward_calls": len(audit.rows), "all_precharged": all(r["precharged"] for r in audit.rows),
                "matches_counter": audit.observed_raw_ef == counter.counters.raw_ef
                    and len(audit.rows) == counter.counters.forward_calls}
        engine.close()
        proof.update(counters=asdict(counter.counters), wall_seconds=time.monotonic() - begin)
        save_json(output / "result.json", proof)
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-execution-preflight", action="store_true", required=True)
    args = parser.parse_args()
    proof = cpu_execution_preflight(args.checkpoint, args.output)
    print(json.dumps({"status": proof["status"], "counters": proof["counters"],
                      "wall_seconds": proof["wall_seconds"]}))


if __name__ == "__main__":
    main()
