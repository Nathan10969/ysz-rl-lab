"""CPU diagnostic contracts, using explicitly synthetic E/F for hook tests."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from ase import Atoms
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import claude08_env as env

try:
    diagnostic = importlib.import_module("review10_652_cpu")
except ModuleNotFoundError as exc:
    if exc.name != "review10_652_cpu":
        raise
    diagnostic = None


class FakeModel(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.calls = 0
        self.fail = fail

    def forward(self, payload, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic failure")
        return {"energy": torch.tensor([1.], dtype=torch.float64),
                "forces": torch.zeros((2, 3), dtype=torch.float64)}


class FakeBatch:
    def __init__(self, value=1.):
        self.value = torch.tensor([value])

    def clone(self):
        clone = FakeBatch()
        clone.value = self.value.clone()
        return clone


class DiagnosticContract(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(diagnostic, "Bounded 652 diagnostic implementation is required")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_only_selected_oxygen_moves_in_minimum_image_direction(self):
        parent = Atoms("O2", positions=[[9.8, 0, 0], [.2, 0, 0]], cell=[10, 10, 10], pbc=True)
        atoms = Atoms("O3", positions=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cell=parent.cell, pbc=True)
        plus, minus, direction = diagnostic.directional_inputs(atoms, parent, atom_index=1,
                                                               source=0, target=1)
        np.testing.assert_allclose(direction, [1, 0, 0], atol=1e-14)
        for candidate, sign in [(plus, 1), (minus, -1)]:
            delta = candidate.positions - atoms.positions
            np.testing.assert_allclose(delta[[0, 2]], 0)
            np.testing.assert_allclose(delta[1], [.1 * sign, 0, 0])
        np.testing.assert_allclose(atoms.positions[1], [4, 5, 6])

    def test_gaussian_is_seeded_raw_and_reused_across_inputs(self):
        a = Atoms("H2", positions=np.zeros((2, 3)))
        b = a.copy(); b.positions += 5
        left = diagnostic.seeded_displacement(a, seed=652, sigma=.02)
        right = diagnostic.seeded_displacement(b, seed=652, sigma=.02)
        expected = np.random.default_rng(652).normal(0, .02, (2, 3))
        np.testing.assert_allclose(left.positions - a.positions, expected)
        np.testing.assert_allclose(right.positions - b.positions, expected)
        self.assertFalse(np.allclose(expected.mean(axis=0), 0))

    def test_registered_matrix_is_exactly_two_directional_and_six_optimizer_cases(self):
        plan = diagnostic.matrix_definitions()
        self.assertEqual(len(plan), 8)
        self.assertEqual([row["optimizer"] for row in plan[:2]], ["BFGS", "BFGS"])
        self.assertEqual([(row["record_id"], row["optimizer"]) for row in plan[2:]],
                         [(rid, name) for rid in (31, 30) for name in ("BFGS", "LBFGS", "FIRE")])
        self.assertTrue(all(row["steps"] == 300 and row["fmax"] == .03 and row["maxstep"] == .1
                            for row in plan))

    def probe(self, model, cap=10):
        meta = type("Metadata", (), {"energy_units_to_eV": 1., "length_units_to_A": 1.})()
        counter = env.RawEFCounter(env.Counters(), env.BudgetLimits(max_raw_ef=cap))
        probe = diagnostic.EFProbe(model, meta, Path(self.tmp.name), counter=counter)
        self.addCleanup(probe.close)
        probe.begin_case("synthetic")
        return probe, counter

    def test_failed_actual_forward_is_precharged_and_preserved(self):
        model = FakeModel(fail=True)
        probe, counter = self.probe(model)
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            probe.evaluate(Atoms("H2"), {"ptr": torch.tensor([0, 2])})
        self.assertEqual((model.calls, counter.counters.raw_ef, counter.counters.forward_calls), (1, 1, 1))
        self.assertEqual(probe.attempts[0]["status"], "failed")

    def test_cap_blocks_real_forward_and_graph_mismatch_is_not_charged(self):
        model = FakeModel()
        probe, counter = self.probe(model, cap=0)
        with self.assertRaises(env.BudgetExceeded):
            probe.evaluate(Atoms("H2"), {"ptr": torch.tensor([0, 2])})
        self.assertEqual((model.calls, counter.counters.raw_ef), (0, 0))
        with self.assertRaisesRegex(RuntimeError, "singleton"):
            probe.evaluate(Atoms("H2"), {"ptr": torch.tensor([0, 1, 2])})
        self.assertEqual(model.calls, 0)

    def test_success_logs_full_real_ef_and_forward_timing(self):
        model = FakeModel()
        probe, counter = self.probe(model)
        energy, forces = probe.evaluate(Atoms("H2"), {"ptr": torch.tensor([0, 2])})
        self.assertEqual(energy, 1.)
        np.testing.assert_array_equal(forces, np.zeros((2, 3)))
        self.assertEqual(len(probe.attempts), 1)
        self.assertGreaterEqual(probe.timings["synthetic"]["forward_seconds"], 0)
        self.assertEqual(probe.latest["synthetic"]["energy_eV"], 1.)
        self.assertEqual(counter.counters.raw_ef, 1)

    def test_timing_records_failed_operations_and_does_not_call_twice(self):
        values = iter([2., 5.])
        totals = {}
        def failure():
            raise RuntimeError("one failed operation")
        with self.assertRaises(RuntimeError):
            diagnostic.timed_call(totals, "graph_seconds", failure, clock=lambda: next(values))
        self.assertEqual(totals, {"graph_seconds": 3.})

    def test_exact_graph_cache_clones_and_invalidates_any_position_change(self):
        calls = []
        def build(atoms):
            calls.append(atoms.positions.copy())
            return FakeBatch(float(len(calls)))
        cache = diagnostic.ExactGraphCache(build)
        atoms = Atoms("H2", positions=np.zeros((2, 3)))
        first = cache.get(atoms)
        first.value[:] = 99
        same = cache.get(atoms.copy())
        self.assertEqual(float(same.value[0]), 1.)
        atoms.positions[0, 0] += 1e-12
        changed = cache.get(atoms)
        self.assertEqual(float(changed.value[0]), 2.)
        self.assertEqual((cache.hits, cache.misses), (1, 2))

    def test_real_parent_spacegroup_metadata_has_stable_exact_geometry_key(self):
        atoms = env._parent().copy()
        self.assertIn("spacegroup", atoms.info)
        self.assertEqual(diagnostic.geometry_key(atoms), diagnostic.geometry_key(atoms.copy()))
        changed = atoms.copy()
        changed.info["unit_cell"] = "metadata changed"
        self.assertNotEqual(diagnostic.geometry_key(atoms), diagnostic.geometry_key(changed))


if __name__ == "__main__":
    unittest.main()
