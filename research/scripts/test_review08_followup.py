import importlib
import unittest
import tempfile
import json
import hashlib
from pathlib import Path

try:
    follow = importlib.import_module('review08_followup')
except ModuleNotFoundError:
    follow = None


class PoolTests(unittest.TestCase):
    def test_recovery_reuses_only_unchanged_independently_verified_terminal_outputs(self):
        self.assertTrue(callable(getattr(follow,'reusable_outputs',None)))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);out=p/'outputs/mace_648';out.mkdir(parents=True)
            f=out/'result.json';f.write_text('{"status":"completed"}')
            proof={'independently_verified':True,'job_id':'648','raw_sha256':{
                str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest()}}
            (p/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(proof))
            self.assertTrue(follow.reusable_outputs(p,'648'))
            f.write_text('changed')
            self.assertFalse(follow.reusable_outputs(p,'648'))

    def combine(self, groups, conditional=2):
        self.assertIsNotNone(follow, 'Supplemental parent-only pool reconciliation is required')
        original = {'jobs': [{'id': 'sft', 'slurm_job_id': '633', 'actual_card_hours': 35/3600},
                             {'id': 'failed', 'slurm_job_id': '635', 'actual_card_hours': .0025}],
                    'accounting_complete': True}
        return follow.combined_budget(original, groups, conditional)

    def test_failed_original_is_charged_and_pending_retry_and_grpo_reserved(self):
        b = self.combine([{'jobs': [{'id': 'retry', 'slurm_job_id': '700', 'state': 'PENDING',
                                    'hard_gpu_hours': .75, 'actual_card_hours': 0}],
                           'accounting_complete': True}])
        self.assertAlmostEqual(b['actual_card_hours'], 44/3600)
        self.assertAlmostEqual(b['actual_plus_remaining_caps'], 44/3600+.75+2)

    def test_unknown_preserves_known_spend_and_full_remaining_reservation(self):
        b = self.combine([{'jobs': [{'id': 'retry', 'slurm_job_id': '700', 'state': 'ACCOUNTING_UNKNOWN',
                                    'hard_gpu_hours': .75, 'actual_card_hours': .1}],
                           'accounting_complete': False}], 0)
        self.assertFalse(b['accounting_complete'])
        self.assertEqual(b['accounting_unknown_job_ids'],['700'])
        self.assertAlmostEqual(b['actual_plus_remaining_caps'], 44/3600+.75)

    def test_successful_retry_replaces_its_reservation_and_duplicate_id_rejected(self):
        g = {'jobs': [{'id': 'retry', 'slurm_job_id': '700', 'state': 'COMPLETED',
                       'hard_gpu_hours': .75, 'actual_card_hours': .05}], 'accounting_complete': True}
        self.assertAlmostEqual(self.combine([g])['actual_plus_remaining_caps'], 44/3600+.05+2)
        with self.assertRaises(ValueError): self.combine([g, g])


if __name__ == '__main__': unittest.main()
