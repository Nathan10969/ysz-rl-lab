"""Pool arithmetic protects reservations, distinct parents and the fixed package."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review10_control import aggregate_budget


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.base = {'jobs': [{'id': 'old', 'slurm_job_id': '650', 'actual_card_hours': .04}],
                     'actual_card_hours': .04, 'accounting_complete': True,
                     'accounting_unknown_job_ids': [], 'remaining_caps': 0}
        self.auth = {'package_cap_card_hours': 6,
                     'stages': [{'stage': x, 'hard_card_hours': 2, 'state': 'HELD'}
                                for x in ('teacher', 'sft', 'grpo')]}
        self.groups = {s['stage']: {'jobs': [], 'accounting_complete': True}
                       for s in self.auth['stages']}

    def test_all_unsubmitted_stages_stay_reserved_without_billing(self):
        root, package = aggregate_budget(self.base, self.auth, self.groups)
        self.assertEqual(package['remaining_caps'], 6)
        self.assertAlmostEqual(root['actual_plus_remaining_caps'], 6.04)
        self.assertAlmostEqual(root['unallocated_gpu_hours'], 3.96)
        self.assertEqual(len(self.base['jobs']), 1)

    def test_known_terminal_replaces_just_its_reservation(self):
        self.groups['teacher'] = {'accounting_complete': True,
            'jobs': [{'slurm_job_id': '701', 'state': 'COMPLETED', 'actual_card_hours': .3,
                      'hard_gpu_hours': 2, 'gpus': 1, 'allocation_gpu_count': 1}]}
        root, package = aggregate_budget(self.base, self.auth, self.groups)
        self.assertEqual(package['remaining_caps'], 4)
        self.assertAlmostEqual(root['actual_card_hours'], .34)
        self.assertEqual(len(root['jobs']), 2)

    def test_unknown_terminal_bill_keeps_entire_unspent_cap(self):
        self.groups['teacher'] = {'accounting_complete': False,
            'accounting_unknown_job_ids': ['701'],
            'jobs': [{'slurm_job_id': '701', 'state': 'COMPLETED', 'actual_card_hours': .3,
                      'hard_gpu_hours': 2, 'gpus': 1}]}
        root, package = aggregate_budget(self.base, self.auth, self.groups)
        self.assertFalse(root['accounting_complete'])
        self.assertAlmostEqual(root['actual_plus_remaining_caps'], 6.04)
        self.assertEqual(root['accounting_unknown_job_ids'], ['701'])

    def test_failed_parent_billed_and_explicit_unreleased_stages_freed(self):
        self.groups['teacher'] = {'accounting_complete': True,
            'jobs': [{'slurm_job_id': '701', 'state': 'FAILED', 'actual_card_hours': .3,
                      'hard_gpu_hours': 2, 'gpus': 1, 'allocation_gpu_count': 1}]}
        for s in self.auth['stages'][1:]:
            s['state'] = 'NOT_RELEASED_UPSTREAM_FAILURE'
        root, package = aggregate_budget(self.base, self.auth, self.groups)
        self.assertEqual(package['remaining_caps'], 0)
        self.assertAlmostEqual(root['actual_card_hours'], .34)

    def test_unreleased_flag_cannot_hide_already_submitted_parent(self):
        self.auth['stages'][0]['state'] = 'NOT_RELEASED_UPSTREAM_FAILURE'
        self.groups['teacher'] = {'accounting_complete': True,
            'jobs': [{'slurm_job_id': '701', 'state': 'PENDING', 'actual_card_hours': 0,
                      'hard_gpu_hours': 2, 'gpus': 1}]}
        with self.assertRaises(ValueError):
            aggregate_budget(self.base, self.auth, self.groups)

    def test_double_parent_or_resource_anomaly_rejected(self):
        self.groups['teacher'] = {'accounting_complete': True,
            'jobs': [{'slurm_job_id': '650', 'state': 'COMPLETED', 'actual_card_hours': .2,
                      'hard_gpu_hours': 2, 'gpus': 1}]}
        with self.assertRaises(ValueError):
            aggregate_budget(self.base, self.auth, self.groups)
        self.groups['teacher']['jobs'][0].update(slurm_job_id='701', allocation_gpu_count=2)
        with self.assertRaises(ValueError):
            aggregate_budget(self.base, self.auth, self.groups)

    def test_cap_and_nonfinite_costs_rejected(self):
        for amount in (2.01, float('nan'), -1):
            self.groups['teacher'] = {'accounting_complete': True,
                'jobs': [{'slurm_job_id': '701', 'state': 'COMPLETED', 'actual_card_hours': amount,
                          'hard_gpu_hours': 2, 'gpus': 1}]}
            with self.assertRaises(ValueError):
                aggregate_budget(self.base, self.auth, self.groups)

    def test_empty_unknown_budget_cannot_release_or_claim_closed_accounting(self):
        self.groups['teacher'].update(accounting_complete=False,
            accounting_unknown_job_ids=['701'])
        root, _ = aggregate_budget(self.base, self.auth, self.groups)
        self.assertFalse(root['accounting_complete'])
        self.assertEqual(root['remaining_caps'], 6)
        self.assertEqual(root['accounting_unknown_job_ids'], ['701'])
        self.auth['stages'][0]['state'] = 'NOT_RELEASED_UPSTREAM_FAILURE'
        with self.assertRaises(ValueError):
            aggregate_budget(self.base, self.auth, self.groups)

    def test_ledger_parent_missing_from_budget_fails_before_publication(self):
        self.groups['teacher']['submitted_job_ids'] = ['701']
        with self.assertRaises(ValueError):
            aggregate_budget(self.base, self.auth, self.groups)

    def test_uncertain_submission_intent_keeps_reservation_until_reconciled(self):
        self.groups['teacher']['pending_submission'] = True
        root, _ = aggregate_budget(self.base, self.auth, self.groups)
        self.assertFalse(root['accounting_complete'])
        self.assertEqual(root['remaining_caps'], 6)
        self.auth['stages'][0]['state'] = 'NOT_RELEASED_UPSTREAM_FAILURE'
        with self.assertRaises(ValueError):
            aggregate_budget(self.base, self.auth, self.groups)

    def test_existing_parent_cannot_coexist_with_another_submission_intent(self):
        self.groups['teacher'] = {'accounting_complete': True, 'pending_submission': True,
            'jobs': [{'slurm_job_id': '701', 'state': 'COMPLETED', 'actual_card_hours': .3,
                      'hard_gpu_hours': 2, 'gpus': 1}]}
        with self.assertRaises(ValueError):
            aggregate_budget(self.base, self.auth, self.groups)

    def test_explicit_unknown_list_overrides_conflicting_complete_flag(self):
        self.groups['teacher'] = {'accounting_complete': True,
            'accounting_unknown_job_ids': ['701'],
            'jobs': [{'slurm_job_id': '701', 'state': 'COMPLETED', 'actual_card_hours': .3,
                      'hard_gpu_hours': 2, 'gpus': 1}]}
        root, _ = aggregate_budget(self.base, self.auth, self.groups)
        self.assertFalse(root['accounting_complete'])
        self.assertAlmostEqual(root['remaining_caps'], 5.7)


if __name__ == '__main__':
    unittest.main()
