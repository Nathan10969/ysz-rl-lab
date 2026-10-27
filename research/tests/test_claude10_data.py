"""Small CPU diagnostic fixtures, never collected physical data."""
import importlib
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import claude08_env as env
from claude08_data import PhysicalReceipt
from calibration import make_state
try:
    data = importlib.import_module('claude10_data')
except ModuleNotFoundError as exc:
    if exc.name != 'claude10_data': raise
    data = None


class DataTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(data, 'Review10 data implementation is required')

    def test_serialized_physical_receipt_is_recomputed_and_forgery_rejected(self):
        request = env.build_request(tuple(map(int, make_state(711, 3, 16).numbers)))
        receipt = PhysicalReceipt.verify(request, env.PhysicsResult(request.atoms, -1., np.zeros((316, 3)), True),
                                         relax_actual=1, raw_ef=2)
        payload = data.serialize_receipt(receipt)
        self.assertEqual(data.deserialize_receipt(payload).summary(), receipt.summary())
        for change in ['energy_eV', 'geometry_hash', 'request_key']:
            forged = json.loads(json.dumps(payload))
            forged['summary'][change] = 'forged'
            with self.assertRaises(ValueError): data.deserialize_receipt(forged)
        payload['result']['converged'] = False
        with self.assertRaises(ValueError): data.deserialize_receipt(payload)

    def test_unscored_rows_are_distinct_balanced_and_exclude_canonical_families(self):
        forbidden = {env.occupancy_key(tuple(map(int, make_state(712, 3, 16).numbers)))}
        rows = data.distinct_unscored_rows(6, forbidden, rng=np.random.default_rng(19), with_actions=True)
        self.assertEqual(len({r['canonical_key'] for r in rows}), 6)
        self.assertFalse({r['canonical_key'] for r in rows} & forbidden)
        self.assertEqual(sum(r['action']['op'] == 'swap_cations' for r in rows), 3)
        for row in rows:
            observation = json.loads(row['messages'][-1]['content'])
            self.assertIsNone(observation['energy_eV'])
            self.assertIsNone(observation['best_energy_eV'])
            self.assertEqual(row['physical_calls'], 0)
            self.assertTrue(row['unscored'])
            self.assertNotIn('physical_receipt', row)
            env.apply_action(row['occupancy'], env.parse_action(json.dumps(row['action'])))
            self.assertEqual(row['messages'], env.encode_prompt_v2(data.unscored_state(row['occupancy'])))

    def test_failed_raw_nonfinite_result_remains_json_serializable(self):
        request = env.build_request(tuple(map(int, make_state(714, 3, 16).numbers)))
        result = env.PhysicsResult(request.atoms, float('nan'), np.full((316,3), float('inf')), False)
        serialized = data.serialize_result(result)
        text = json.dumps(serialized, allow_nan=False)
        self.assertIn('NaN', text)
        self.assertIn('Infinity', text)

    def test_late_result_write_changes_durable_completed_status_to_failure(self):
        self.assertTrue(hasattr(data,'persist_result'), 'Bounded durable result finalizer required')
        with tempfile.TemporaryDirectory() as tmp:
            now = [6599.]
            actual_write = data.write_json
            def slow_write(path, value):
                actual_write(path,value)
                now[0] += 2.
            result = {'status':'completed','adapter_release_allowed':True}
            with patch.object(data,'write_json',side_effect=slow_write):
                overrun = data.persist_result(Path(tmp)/'result.json',result,started_at=0.,clock=lambda:now[0])
            self.assertTrue(overrun)
            saved = json.loads((Path(tmp)/'result.json').read_text())
            self.assertEqual(saved['status'],'failed')
            self.assertFalse(saved['adapter_release_allowed'])
            self.assertGreaterEqual(saved['wall_seconds'],6600.)

    def test_claimed_completed_teacher_without_atoms_and_receipts_is_rejected(self):
        with self.assertRaises(ValueError):
            data.verify_teacher_result({'status':'completed', 'trajectories':[{'complete':True}] * 16})

    def test_overlapping_trajectories_remain_in_one_split(self):
        trajectories = [{'terminal_families':['a','b']}, {'terminal_families':['b','c']},
                        {'terminal_families':['d']}, {'terminal_families':['e']}]
        train, valid = data.split_terminal_families(trajectories, seed=42)
        self.assertFalse(set().union(*(set(r['terminal_families']) for r in train)) &
                         set().union(*(set(r['terminal_families']) for r in valid)))


if __name__ == '__main__': unittest.main()
