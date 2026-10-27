"""Offline contract tests; quadratic E/F are explicitly synthetic test inputs."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import claude08_env as env

try:
    physics = importlib.import_module("claude10_physics")
except ModuleNotFoundError as exc:
    if exc.name != "claude10_physics":
        raise
    physics = None


class QuadraticCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        delta = atoms.positions - atoms.info["target"]
        self.results = {"energy": float(.5 * (delta ** 2).sum()), "forces": -delta}


class QuadraticModel(torch.nn.Module):
    def __init__(self, *, fail=False):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.sizes = []
        self.fail = fail

    def forward(self, payload, **kwargs):
        group = payload["group"]
        self.sizes.append(len(group))
        if self.fail:
            raise RuntimeError("synthetic forward failure")
        delta = [cell.positions - cell.info["target"] for cell in group]
        return {"energy": torch.tensor([.5 * (x ** 2).sum() for x in delta], dtype=torch.float64),
                "forces": torch.tensor(-np.concatenate(delta), dtype=torch.float64)}


class FakeBatch:
    def __init__(self, group):
        self.group = group

    def to_dict(self):
        return {"group": self.group,
                "ptr": torch.tensor(np.cumsum([0] + [len(cell) for cell in self.group]))}


class BatchPhysicsContract(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(physics, "Review10 batched physics implementation is required")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def request(self, index=0, offset=.5):
        atoms = Atoms("H2", positions=np.zeros((2, 3)), cell=np.eye(3) * 10, pbc=True)
        atoms.info["target"] = np.full((2, 3), offset)
        return env.PhysicsRequest((), atoms, env.RelaxationSpec(), f"test-lane-{index}")

    def counter(self, cap=1000):
        return env.RawEFCounter(env.Counters(), env.BudgetLimits(max_raw_ef=cap))

    def engine(self, model=None, actor=None):
        model = QuadraticModel() if model is None else model
        calc = type("CalculatorMetadata", (), {"models": [model], "energy_units_to_eV": 1.,
                    "length_units_to_A": 1., "head": "Default"})()
        self.enterContext(patch.object(physics, "_load_calculator", return_value=calc))
        self.enterContext(patch.object(physics, "build_batch", side_effect=lambda calc, group, device: FakeBatch(group)))
        instance = physics.BatchMACEPhysics(ROOT / "models/mace-small-density-agnesi-stress.model",
                    Path(self.tmp.name), device="cpu", actor=actor)
        return instance, model

    def test_batched_bfgs_matches_independent_ase_and_singleton_runs(self):
        requests = [self.request(i, offset) for i, offset in enumerate((.1, .4, .7, 1.))]
        engine, model = self.engine()
        batch = engine.relax_many(requests, self.counter())
        self.assertIn(4, model.sizes)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        for request, result in zip(requests, batch):
            atoms = request.atoms.copy()
            atoms.calc = QuadraticCalculator()
            optimizer = BFGS(atoms, logfile=None, maxstep=.1)
            optimizer.run(fmax=.03, steps=300)
            np.testing.assert_allclose(result.atoms.positions, atoms.positions, rtol=0, atol=1e-12)
            self.assertEqual(result.evidence["nsteps"], optimizer.nsteps)
            self.assertTrue(result.converged)
            single = engine.relax_many([request], self.counter())[0]
            np.testing.assert_allclose(result.forces, single.forces, rtol=0, atol=1e-12)
            self.assertAlmostEqual(result.energy, single.energy, places=12)
            self.assertEqual(result.evidence["raw_ef_attempts"], result.evidence["nsteps"] + 1)

    def test_active_set_shrinks_without_padding_and_lanes_stop_independently(self):
        requests = [self.request(i, offset) for i, offset in enumerate((0., .04, .2, .5))]
        engine, model = self.engine()
        counter = self.counter()
        results = engine.relax_many(requests, counter)
        self.assertEqual(model.sizes[0], 4)
        self.assertEqual(model.sizes[1], 3)
        self.assertIn(2, model.sizes)
        self.assertIn(1, model.sizes)
        self.assertEqual(results[0].evidence["nsteps"], 0)
        self.assertEqual(results[0].evidence["raw_ef_attempts"], 1)
        self.assertEqual(counter.counters.raw_ef, sum(model.sizes))
        self.assertEqual(counter.counters.raw_ef, sum(r.evidence["raw_ef_attempts"] for r in results))
        self.assertEqual(counter.counters.forward_calls, len(model.sizes))
        self.assertEqual([r.evidence["lane"] for r in results], list(range(4)))

    def test_hard_cap_rejects_whole_forward_before_model_execution(self):
        engine, model = self.engine()
        counter = self.counter(cap=3)
        with self.assertRaises(env.BudgetExceeded):
            engine.relax_many([self.request(i) for i in range(4)], counter)
        self.assertEqual(model.sizes, [])
        self.assertEqual(counter.counters.raw_ef, 0)

    def test_mid_relaxation_cap_keeps_first_charge_and_stops_next_forward(self):
        engine, model = self.engine()
        counter = self.counter(cap=4)
        with self.assertRaises(env.BudgetExceeded):
            engine.relax_many([self.request(i) for i in range(4)], counter)
        self.assertEqual(model.sizes, [4])
        self.assertEqual(counter.counters.raw_ef, 4)
        self.assertEqual([r["raw_ef_attempts"] for r in engine.last_evidence], [1] * 4)
        self.assertTrue(all(r["status"] == "budget_exhausted" for r in engine.last_evidence))

    def test_cuda_rejects_short_relaxation_before_load_and_uses_gpu_guard(self):
        engine, _ = self.engine()
        engine.device = "cuda"
        with self.assertRaises(ValueError):
            engine.relax_many([self.request()], self.counter(), max_steps=1)
        with patch("smoke_grpo08.require_single_gpu_job", side_effect=RuntimeError("missing allocation")) as guard:
            with self.assertRaisesRegex(physics.BatchPhysicsError, "missing allocation"):
                engine.relax_many([self.request()], self.counter())
            guard.assert_called_once()

    def test_actual_forward_audit_checks_graph_count_against_precharged_counter(self):
        counter = self.counter()
        self.assertTrue(hasattr(physics, "ForwardAudit"), "Independent actual-forward audit is required")
        audit = physics.ForwardAudit(counter)
        payload = {"ptr": torch.tensor([0, 2, 4, 6, 8])}
        with self.assertRaisesRegex(RuntimeError, "precharged"):
            audit(None, (payload,), {})
        counter.before_forward(4)
        audit(None, (payload,), {})
        self.assertEqual(audit.observed_raw_ef, 4)
        self.assertEqual(len(audit.rows), 1)
        counter.before_forward(1)
        audit(None, ({"ptr": torch.tensor([0, 2])},), {})
        self.assertEqual(audit.observed_raw_ef, 5)

    def test_failed_forward_retains_attempts_and_has_no_fabricated_result(self):
        engine, model = self.engine(QuadraticModel(fail=True))
        counter = self.counter()
        engine._ensure(counter)
        observed = []
        hook = model.register_forward_pre_hook(lambda model, args: observed.append(
            (int(args[0]["ptr"].numel() - 1), counter.counters.raw_ef, counter.counters.forward_calls)))
        self.addCleanup(hook.remove)
        with self.assertRaises(physics.BatchPhysicsError) as caught:
            engine.relax_many([self.request(i) for i in range(4)], counter)
        self.assertEqual(counter.counters.raw_ef, 4)
        self.assertEqual(counter.counters.forward_calls, 1)
        self.assertEqual(model.sizes, [4])
        self.assertEqual(observed, [(4, 4, 1)])
        self.assertEqual([r["raw_ef_attempts"] for r in caught.exception.evidence], [1] * 4)
        self.assertTrue(all("energy_eV" not in r for r in caught.exception.evidence))
        self.assertTrue(all(r["status"] == "failed" for r in caught.exception.evidence))

    def test_production_actual_graph_mismatch_stops_before_forward_and_charge(self):
        engine, model = self.engine()
        counter = self.counter()
        request = self.request()
        wrong_batch = FakeBatch([request.atoms, request.atoms.copy()])
        with patch.object(physics, "build_batch", return_value=wrong_batch):
            with self.assertRaisesRegex(physics.BatchPhysicsError, "graph.*request"):
                engine.single_points([request], counter)
        self.assertEqual(model.sizes, [])
        self.assertEqual((counter.counters.raw_ef, counter.counters.forward_calls), (0, 0))
        self.assertEqual(engine.last_evidence[0]["raw_ef_attempts"], 0)

    def test_production_model_rejects_forward_without_counter_context(self):
        engine, model = self.engine()
        counter = self.counter()
        engine._ensure(counter)
        payload = FakeBatch([self.request().atoms]).to_dict()
        with self.assertRaisesRegex(RuntimeError, "context"):
            model(payload)
        self.assertEqual(model.sizes, [])
        self.assertEqual(counter.counters.raw_ef, 0)

    def test_cpu_one_step_preserves_unconverged_cost_and_cannot_verify_terminal(self):
        engine, _ = self.engine()
        counter = self.counter()
        result = engine.relax_many([self.request()], counter, max_steps=1)[0]
        self.assertFalse(result.converged)
        self.assertFalse(result.evidence["formal_receipt"])
        self.assertEqual(result.evidence["nsteps"], 1)
        self.assertEqual(result.evidence["raw_ef_attempts"], 2)
        self.assertEqual(counter.counters.raw_ef, 2)
        with self.assertRaisesRegex(ValueError, "did not converge"):
            env.verify_result(result, env.RelaxationSpec())

    def test_request_batch_size_hash_dtype_and_step_bound_checked_before_load(self):
        engine, model = self.engine()
        request = self.request()
        bad_spec = type("BadSpec", (), {"dtype": "float32", "model_sha": env.MODEL_SHA})()
        bad = env.PhysicsRequest((), request.atoms, bad_spec, request.key)
        for requests in ([], [request] * 5, [bad]):
            with self.subTest(size=len(requests)), self.assertRaises(ValueError):
                engine.relax_many(requests, self.counter())
        with self.assertRaises(ValueError):
            engine.relax_many([request], self.counter(), max_steps=2)
        wrong_hash = env.PhysicsRequest((), request.atoms, env.RelaxationSpec(model_sha="wrong"), request.key)
        with self.assertRaises(ValueError):
            engine.single_points([wrong_hash], self.counter())
        self.assertEqual(model.sizes, [])

    def test_checkpoint_mismatch_blocks_before_loading(self):
        path = Path(self.tmp.name) / "wrong.model"
        path.write_bytes(b"wrong frozen checkpoint")
        with patch.object(physics, "_load_calculator") as load:
            with self.assertRaisesRegex(ValueError, "SHA"):
                physics.BatchMACEPhysics(path, Path(self.tmp.name) / "run", device="cpu")
            load.assert_not_called()

    def test_cpu_path_never_reads_cuda_and_actor_offloads_before_load(self):
        events = []
        class Actor:
            def to_device(self, device, ledger):
                ledger.check()
                events.append(device)
        engine, _ = self.engine(actor=Actor())
        class ForbiddenCUDA:
            def __getattr__(self, name):
                raise AssertionError("CPU path touched torch.cuda." + name)
        with patch.object(torch, "cuda", ForbiddenCUDA()):
            results = engine.single_points([self.request()], self.counter())
        self.assertEqual(events, ["cpu"])
        self.assertEqual(results[0].evidence["status"], "single_point")
        self.assertFalse(results[0].evidence["formal_receipt"])

    def test_native_model_unit_conversions_are_applied_to_energy_and_forces(self):
        engine, _ = self.engine()
        results = engine.single_points([self.request(offset=.5)], self.counter())
        raw_energy = .5 * 6 * .5 ** 2
        self.assertAlmostEqual(results[0].energy, raw_energy)
        np.testing.assert_allclose(results[0].forces, .5)
        engine.calc.energy_units_to_eV = 2.
        engine.calc.length_units_to_A = 4.
        scaled = engine.single_points([self.request(offset=.5)], self.counter())[0]
        self.assertAlmostEqual(scaled.energy, raw_energy * 2.)
        np.testing.assert_allclose(scaled.forces, .25)


if __name__ == "__main__":
    unittest.main()
