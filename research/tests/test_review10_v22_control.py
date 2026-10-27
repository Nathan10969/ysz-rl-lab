"""Recovery fees stay in the original pool, with the failed652 parent retained."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    control = importlib.import_module('review10_v22_control')
except ModuleNotFoundError as exc:
    if exc.name != 'review10_v22_control':
        raise
    control = None


CAPS = {'teacher': 2., 'sft': 1.85, 'grpo': 2.}
SECONDS = {'633': 35, '635': 9, '648': 36, '650': 72, '652': 507, '656': 153}


def authorization():
    return dict(schema='review10-v22-authorization-v1', direct_user_confirmation=True,
        state='CPU_DIAGNOSTIC_AND_PROTOCOL_REVIEW', package_total_cap_including652_656_card_hours=6.5,
        historical_package_actual_card_hours=660/3600, historical_root_actual_card_hours=812/3600,
        root_pool_card_hours=10, max_concurrent_GPU_jobs=1, max_new_teacher_submissions=1,
        automatic_retry=False, formal_training_allowed=False, new_DFT_allowed=False,
        stage_caps_card_hours=copy.deepcopy(CAPS))


def baseline():
    rows = [dict(id='old_'+sid, slurm_job_id=sid, actual_card_hours=seconds/3600,
                 state='FAILED' if sid in ('635', '652', '656') else 'COMPLETED',
                 gpus=1, allocation_gpu_count=1, unexpected_gpu_allocation=False,
                 hard_cap_exceeded=False,
                 sacct=dict(job_id=sid, elapsed_seconds=seconds, state='FAILED' if sid in ('635','652','656')
                            else 'COMPLETED', alloc_tres='cpu=6,mem=90G,gres/gpu=1,gres/gpu:a100=1'))
            for sid, seconds in SECONDS.items()]
    return dict(jobs=rows, authorized_card_hours=10, actual_card_hours=812/3600,
                accounting_complete=True, accounting_unknown_job_ids=[], remaining_caps=0)


def groups():
    return {stage:dict(jobs=[], submitted_job_ids=[], pending_submission=False,
            authorized_card_hours=cap, authorized_gpu_hours=cap, actual_card_hours=0,
            accounting_complete=True, accounting_unknown_job_ids=[], state='HELD')
            for stage, cap in CAPS.items()}


def parent(stage='teacher', sid='701', seconds=360, state='COMPLETED', allocation=1):
    return dict(id='new_'+stage, slurm_job_id=sid, state=state,
        actual_card_hours=seconds*allocation/3600, hard_gpu_hours=CAPS[stage], gpus=1,
        cpu_cores=6, memory_GiB=90, allocation_gpu_count=allocation,
        unexpected_gpu_allocation=allocation != 1, hard_cap_exceeded=False,
        sacct=dict(job_id=sid, state=state, elapsed_seconds=seconds,
                   alloc_tres='' if allocation == 0 else 'cpu=6,mem=90G,gres/gpu=1,gres/gpu:a100=1',
                   exit_code='0:0'))


class RecoveryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(control, 'New recovery controller required')
        self.base, self.auth, self.groups = baseline(), authorization(), groups()

    def submitted(self, stage='teacher', **kwargs):
        job = parent(stage, **kwargs)
        self.groups[stage].update(jobs=[job], submitted_job_ids=[job['slurm_job_id']])
        return job

    def test_unsubmitted_three_caps_include652_656_inside_additional6p5(self):
        original = copy.deepcopy((self.base, self.auth, self.groups))
        root, package = control.aggregate_budget(self.base, self.auth, self.groups)
        self.assertAlmostEqual(root['remaining_caps'], 5.85)
        self.assertAlmostEqual(root['actual_plus_remaining_caps'], 812/3600 + 5.85)
        self.assertAlmostEqual(package['actual_card_hours'], 660/3600)
        self.assertAlmostEqual(package['actual_plus_remaining_caps'], 660/3600 + 5.85)
        self.assertLess(package['actual_plus_remaining_caps'], 6.5)
        self.assertEqual(len(root['jobs']), 6)
        self.assertEqual((self.base, self.auth, self.groups), original)

    def test_known_failed_parent_keeps_cost_and_releases_only_its_remaining_cap(self):
        self.submitted(state='FAILED', seconds=180)
        root, package = control.aggregate_budget(self.base, self.auth, self.groups)
        self.assertAlmostEqual(root['actual_card_hours'], 812/3600 + .05)
        self.assertAlmostEqual(root['remaining_caps'], 3.85)
        self.assertAlmostEqual(package['actual_card_hours'], 660/3600 + .05)
        self.assertEqual([str(x['slurm_job_id']) for x in root['jobs']].count('652'), 1)

    def test_running_parent_retains_unspent_part_of_cap(self):
        self.submitted(state='RUNNING')
        root, _ = control.aggregate_budget(self.base, self.auth, self.groups)
        self.assertAlmostEqual(root['remaining_caps'], 5.75)
        self.assertAlmostEqual(root['actual_plus_remaining_caps'], 812/3600 + 5.85)

    def test_unknown_terminal_bill_retains_reserve_and_known_lower_bound(self):
        self.submitted()
        self.groups['teacher'].update(accounting_complete=False, accounting_unknown_job_ids=['701'])
        root, _ = control.aggregate_budget(self.base, self.auth, self.groups)
        self.assertFalse(root['accounting_complete'])
        self.assertEqual(root['accounting_unknown_job_ids'], ['701'])
        self.assertAlmostEqual(root['actual_card_hours'], 812/3600 + .1)
        self.assertAlmostEqual(root['actual_plus_remaining_caps'], 812/3600 + 5.85)

    def test_unknown_ids_override_inconsistent_complete_flag(self):
        self.submitted()
        self.groups['teacher']['accounting_unknown_job_ids'] = ['701']
        root, _ = control.aggregate_budget(self.base, self.auth, self.groups)
        self.assertFalse(root['accounting_complete'])
        self.assertAlmostEqual(root['remaining_caps'], 5.75)

    def test_unsubmitted_withheld_successors_free_only_known_no_intent_stages(self):
        self.submitted(state='FAILED')
        for stage in ('sft', 'grpo'):
            self.groups[stage]['state'] = 'NOT_RELEASED_UPSTREAM_FAILURE'
        root, package = control.aggregate_budget(self.base, self.auth, self.groups)
        self.assertEqual(root['remaining_caps'], 0)
        self.assertAlmostEqual(package['actual_card_hours'], 660/3600 + .1)
        self.groups['sft']['accounting_complete'] = False
        with self.assertRaises(ValueError):
            control.aggregate_budget(self.base, self.auth, self.groups)

    def test_withheld_cannot_hide_submitted_parent_or_pending_intent(self):
        for submitted in (True, False):
            g = groups()
            g['teacher']['state'] = 'NOT_RELEASED_UPSTREAM_FAILURE'
            if submitted:
                g['teacher'].update(jobs=[parent()], submitted_job_ids=['701'])
            else:
                g['teacher']['pending_submission'] = True
            with self.assertRaises(ValueError):
                control.aggregate_budget(self.base, self.auth, g)

    def test_pending_submission_keeps_full_reserve_and_marks_unknown(self):
        self.groups['teacher']['pending_submission'] = True
        root, _ = control.aggregate_budget(self.base, self.auth, self.groups)
        self.assertFalse(root['accounting_complete'])
        self.assertIn('UNRESOLVED_SUBMISSION_teacher', root['accounting_unknown_job_ids'])
        self.assertAlmostEqual(root['remaining_caps'], 5.85)

    def test_ledger_budget_mismatch_or_parent_and_intent_refused(self):
        self.groups['teacher']['submitted_job_ids'] = ['701']
        with self.assertRaises(ValueError):
            control.aggregate_budget(self.base, self.auth, self.groups)
        self.submitted()
        self.groups['teacher']['pending_submission'] = True
        with self.assertRaises(ValueError):
            control.aggregate_budget(self.base, self.auth, self.groups)

    def test_duplicate_historical_or_new_parent_refused(self):
        self.submitted(sid='652')
        with self.assertRaises(ValueError):
            control.aggregate_budget(self.base, self.auth, self.groups)
        self.submitted()
        self.submitted('sft', sid='701')
        with self.assertRaises(ValueError):
            control.aggregate_budget(self.base, self.auth, self.groups)

    def test_one_parent_per_stage_and_fixed_caps(self):
        self.submitted()
        self.groups['teacher']['jobs'].append(parent(sid='702'))
        with self.assertRaises(ValueError):
            control.aggregate_budget(self.base, self.auth, self.groups)
        self.groups = groups()
        self.auth['stage_caps_card_hours']['sft'] = 2
        with self.assertRaises(ValueError):
            control.aggregate_budget(self.base, self.auth, self.groups)

    def test_sft_1p85_cap_cannot_be_treated_as2(self):
        self.submitted('sft', seconds=6661)
        with self.assertRaises(ValueError):
            control.aggregate_budget(self.base, self.auth, self.groups)

    def test_nonfinite_negative_or_bool_amount_refused(self):
        for amount in (float('nan'), float('inf'), -1, True):
            job = self.submitted()
            job['actual_card_hours'] = amount
            with self.assertRaises(ValueError):
                control.aggregate_budget(self.base, self.auth, self.groups)

    def test_pending_zero_GPU_with_zero_elapsed_is_not_allocation_anomaly(self):
        self.submitted(state='PENDING', seconds=0, allocation=0)
        root, _ = control.aggregate_budget(self.base, self.auth, self.groups)
        self.assertTrue(root['accounting_complete'])
        self.assertAlmostEqual(root['remaining_caps'], 5.85)
        self.groups['teacher']['jobs'][0]['state'] = 'RUNNING'
        self.groups['teacher']['jobs'][0]['sacct']['state'] = 'RUNNING'
        with self.assertRaises(ValueError):
            control.aggregate_budget(self.base, self.auth, self.groups)

    def test_real_allocation_resource_or_declared_risk_refused(self):
        for changes in (dict(allocation_gpu_count=2), dict(cpu_cores=7),
                        dict(memory_GiB=91), dict(gpus=2), dict(hard_cap_exceeded=True)):
            job = self.submitted()
            job.update(changes)
            with self.assertRaises(ValueError):
                control.aggregate_budget(self.base, self.auth, self.groups)

    def test_saved_actual_GPU_CPU_and_memory_must_match_declared_allocation(self):
        for tres in ('cpu=6,mem=90G,gres/gpu=2,gres/gpu:a100=2',
                     'cpu=12,mem=90G,gres/gpu=1,gres/gpu:a100=1',
                     'cpu=6,mem=91G,gres/gpu=1,gres/gpu:a100=1'):
            job = self.submitted()
            job['sacct']['alloc_tres'] = tres
            with self.assertRaises(ValueError):
                control.aggregate_budget(self.base, self.auth, self.groups)
        job = self.submitted()
        job['sacct']['alloc_tres'] = 'cpu=6,mem=92160M,gres/gpu=1,gres/gpu:a100=1'
        self.assertTrue(control.aggregate_budget(self.base, self.auth, self.groups)[0]['accounting_complete'])

    def test_empty_or_malformed_intent_marker_cannot_disappear_as_false(self):
        for value in ({}, [], '', 0):
            g = groups()
            g['teacher'].update(state='NOT_RELEASED_UPSTREAM_FAILURE', pending_submission=value)
            with self.assertRaises(ValueError):
                control.aggregate_budget(self.base, self.auth, g)

    def test_missing_or_invalid_authorization_state_fails_before_query_or_write(self):
        for state in (None, '', {}, 0):
            auth = authorization()
            if state is None:
                auth.pop('state')
            else:
                auth['state'] = state
            with self.assertRaises(ValueError):
                control.aggregate_budget(self.base, auth, self.groups)

    def test_baseline_must_retain_all6_verified_original_parents_and_exact_costs(self):
        for mutate in (lambda b:b['jobs'].pop(),
                       lambda b:b.update(accounting_complete=False),
                       lambda b:b['jobs'][-1].update(actual_card_hours=0),
                       lambda b:b.update(actual_card_hours=0)):
            base = baseline()
            mutate(base)
            with self.assertRaises(ValueError):
                control.aggregate_budget(base, self.auth, self.groups)


class RecoveryCLITests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(control, 'New recovery controller required')

    def stage_fixture(self, package):
        (package/'AUTHORIZATION.json').write_text(json.dumps(authorization()))
        (package/'BASELINE_BUDGET.json').write_text(json.dumps(baseline()))
        for stage, group in groups().items():
            path = package/stage
            path.mkdir()
            (path/'BUDGET.json').write_text(json.dumps(group))
            (path/'SUBMISSIONS.json').write_text(json.dumps(dict(jobs=[])))

    def test_publish_is_local_only_and_preserves_unrelated_root_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            package = home/'recovery'
            package.mkdir()
            self.stage_fixture(package)
            (home/'ANALYSIS.json').write_text(json.dumps(dict(old={'keep':True})))
            with patch.object(control, 'HOME', home), patch.object(control, 'PACKAGE', package), \
                 patch.object(control, 'validate_frozen_history') as frozen, \
                 patch.object(control.collect, 'query') as query:
                control.main(['--publish'])
                frozen.assert_called_once()
                query.assert_not_called()
            saved = json.loads((home/'BUDGET.json').read_text())
            self.assertAlmostEqual(saved['actual_plus_remaining_caps'], 812/3600 + 5.85)
            analysis = json.loads((home/'ANALYSIS.json').read_text())
            self.assertEqual(analysis['old'], {'keep':True})
            self.assertIn('review10_v22_recovery', analysis)

    def test_query_failure_is_one_attempt_and_retains_unknown_fee_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            home = package/'teacher'
            home.mkdir()
            job = parent(seconds=0, state='PENDING')
            job.update(remote_output='/remote/training_smoke/claude10/recovery_v22/teacher/outputs/teacher_701', remote_log='/remote/training_smoke/claude10/recovery_v22/teacher/slurm-701.out')
            (home/'SUBMISSIONS.json').write_text(json.dumps(dict(jobs=[job])))
            (home/'JOB_MANIFEST.json').write_text(json.dumps(dict(jobs=[job], remote_smoke_root='/remote/training_smoke/claude10/recovery_v22/teacher')))
            (home/'BUDGET.json').write_text(json.dumps(dict(groups()['teacher'], jobs=[job])))
            with patch.object(control, 'PACKAGE', package), \
                 patch.object(control.collect, 'query', side_effect=OSError('network unavailable')) as query:
                control.recover_stage('teacher')
                query.assert_called_once()
            budget = json.loads((home/'BUDGET.json').read_text())
            self.assertFalse(budget['accounting_complete'])
            self.assertEqual(budget['accounting_unknown_job_ids'], ['701'])
            self.assertEqual(budget['jobs'][0]['actual_card_hours'], 0)
            self.assertEqual(json.loads((home/'COLLECTION_FAILURE.json').read_text())['collection_status'], 'DEFERRED')

    def test_malformed_auth_does_not_query_or_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            (package/'AUTHORIZATION.json').write_text('{}')
            with patch.object(control, 'PACKAGE', package), \
                 patch.object(control.collect, 'query') as query, \
                 patch.object(control, 'atomic_json') as write:
                with self.assertRaises(ValueError):
                    control.main(['--collect'])
                query.assert_not_called()
                write.assert_not_called()

    def test_raw_memory_conflict_is_deferred_before_reconciliation_or_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            home = package/'teacher'
            home.mkdir()
            job = parent(seconds=0, state='PENDING')
            job.update(remote_output='/remote/training_smoke/claude10/recovery_v22/teacher/outputs/teacher_701', remote_log='/remote/training_smoke/claude10/recovery_v22/teacher/slurm-701.out')
            (home/'SUBMISSIONS.json').write_text(json.dumps(dict(jobs=[job])))
            (home/'JOB_MANIFEST.json').write_text(json.dumps(dict(jobs=[job], remote_smoke_root='/remote/training_smoke/claude10/recovery_v22/teacher')))
            (home/'BUDGET.json').write_text(json.dumps(dict(groups()['teacher'], jobs=[job])))
            snapshot = dict(ledger={'ledger_status':'RECONCILED'}, account=dict(code=0,
                stdout='701|RUNNING|36|cpu=6,mem=91G,gres/gpu=1,gres/gpu:a100=1|0:0|6|1\n'))
            with patch.object(control, 'PACKAGE', package), \
                 patch.object(control.collect, 'query', return_value=snapshot), \
                 patch.object(control.collect, 'reconcile_budget') as reconcile, \
                 patch.object(control.subprocess, 'run') as copy:
                control.recover_stage('teacher')
                reconcile.assert_not_called()
                copy.assert_not_called()
            budget = json.loads((home/'BUDGET.json').read_text())
            self.assertFalse(budget['accounting_complete'])
            self.assertTrue(budget['collection_risks'])

    def test_deferred_resource_risk_keeps_previously_published_fees_and_names_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            package = home/'recovery'
            package.mkdir()
            self.stage_fixture(package)
            prior, pack = control.aggregate_budget(baseline(), authorization(), groups())
            (home/'BUDGET.json').write_text(json.dumps(prior))
            (package/'BUDGET.json').write_text(json.dumps(pack))
            stage = package/'teacher'
            job = parent(seconds=0, state='PENDING')
            bad = dict(groups()['teacher'], jobs=[job], accounting_complete=False,
                       accounting_unknown_job_ids=['701'], collection_risks=['observed_allocation_conflict'])
            (stage/'BUDGET.json').write_text(json.dumps(bad))
            (stage/'SUBMISSIONS.json').write_text(json.dumps(dict(jobs=[job])))
            with patch.object(control, 'HOME', home), patch.object(control, 'PACKAGE', package), \
                 patch.object(control, 'validate_frozen_history'), \
                 patch.object(control.collect, 'query') as query:
                with self.assertRaises(ValueError):
                    control.main(['--publish'])
                query.assert_not_called()
            saved = json.loads((home/'BUDGET.json').read_text())
            self.assertEqual(saved['collection_status'], 'DEFERRED')
            self.assertEqual(saved['actual_card_hours'], prior['actual_card_hours'])
            self.assertEqual(saved['remaining_caps'], prior['remaining_caps'])
            self.assertFalse(saved['accounting_complete'])
            self.assertEqual(saved['accounting_unknown_job_ids'], ['701'])

    def test_explicit_collect_can_observe_once_after_prior_resource_risk(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            package = home/'recovery'
            package.mkdir()
            self.stage_fixture(package)
            stage = package/'teacher'
            job = parent(seconds=0, state='PENDING')
            bad = dict(groups()['teacher'], jobs=[job], accounting_complete=False,
                       accounting_unknown_job_ids=['701'], collection_risks=['old_observation'])
            (stage/'BUDGET.json').write_text(json.dumps(bad))
            (stage/'SUBMISSIONS.json').write_text(json.dumps(dict(jobs=[job])))
            def new_observation(name):
                self.assertEqual(name, 'teacher')
                fresh = dict(groups()['teacher'], jobs=[job])
                (stage/'BUDGET.json').write_text(json.dumps(fresh))
            with patch.object(control, 'HOME', home), patch.object(control, 'PACKAGE', package), \
                 patch.object(control, 'validate_frozen_history'), \
                 patch.object(control, 'recover_stage', side_effect=new_observation) as recover:
                control.main(['--collect', '--stage', 'teacher'])
                recover.assert_called_once_with('teacher')
            saved = json.loads((home/'BUDGET.json').read_text())
            self.assertTrue(saved['accounting_complete'])
            self.assertAlmostEqual(saved['remaining_caps'], 5.85)

    def test_fresh_resource_risk_restores_reserve_released_by_verified_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            package = home/'recovery'
            package.mkdir()
            self.stage_fixture(package)
            stage = package/'teacher'
            job = parent(seconds=360, state='COMPLETED')
            job.update(remote_output='/remote/training_smoke/claude10/recovery_v22/teacher/outputs/teacher_701', remote_log='/remote/training_smoke/claude10/recovery_v22/teacher/slurm-701.out')
            saved_groups = groups()
            saved_groups['teacher'].update(jobs=[job], submitted_job_ids=['701'])
            (stage/'BUDGET.json').write_text(json.dumps(saved_groups['teacher']))
            (stage/'SUBMISSIONS.json').write_text(json.dumps(dict(jobs=[job])))
            (stage/'JOB_MANIFEST.json').write_text(json.dumps(dict(jobs=[job], remote_smoke_root='/remote/training_smoke/claude10/recovery_v22/teacher')))
            prior, pack = control.aggregate_budget(baseline(), authorization(), saved_groups)
            self.assertAlmostEqual(prior['remaining_caps'], 3.85)
            (home/'BUDGET.json').write_text(json.dumps(prior))
            (package/'BUDGET.json').write_text(json.dumps(pack))
            snapshot = dict(ledger={'ledger_status':'RECONCILED'}, account=dict(code=0,
                stdout='701|COMPLETED|360|cpu=6,mem=91G,gres/gpu=1,gres/gpu:a100=1|0:0|6|1\n'))
            raw = json.dumps(snapshot)
            def fresh_conflicting_observation(*args):
                (stage/'LATEST.json').write_text(raw)
                return snapshot
            with patch.object(control, 'HOME', home), patch.object(control, 'PACKAGE', package), \
                 patch.object(control, 'validate_frozen_history'), \
                 patch.object(control.collect, 'query', side_effect=fresh_conflicting_observation) as query, \
                 patch.object(control.collect, 'reconcile_budget') as reconcile, \
                 patch.object(control.subprocess, 'run') as copy_output:
                with self.assertRaises(ValueError):
                    control.main(['--collect', '--stage', 'teacher'])
                query.assert_called_once()
                reconcile.assert_not_called()
                copy_output.assert_not_called()
            root = json.loads((home/'BUDGET.json').read_text())
            package_budget = json.loads((package/'BUDGET.json').read_text())
            self.assertEqual((stage/'LATEST.json').read_text(), raw)
            self.assertEqual(root['collection_status'], 'DEFERRED')
            self.assertFalse(root['accounting_complete'])
            self.assertEqual(root['accounting_unknown_job_ids'], ['701'])
            self.assertEqual(root['actual_card_hours'], prior['actual_card_hours'])
            self.assertEqual(package_budget['actual_card_hours'], pack['actual_card_hours'])
            self.assertAlmostEqual(root['review10_recovery_new_actual_card_hours'], .1)
            self.assertAlmostEqual(package_budget['new_actual_card_hours'], .1)
            for budget in (root, package_budget):
                self.assertAlmostEqual(budget['remaining_caps'], 5.75)
                self.assertAlmostEqual(budget['actual_plus_remaining_caps'], budget['actual_card_hours']+5.75)
            self.assertAlmostEqual(package_budget['new_remaining_caps'], 5.75)
            self.assertAlmostEqual(package_budget['package_actual_plus_remaining_caps'], pack['actual_card_hours']+5.75)
            self.assertAlmostEqual(root['review10_reservation_card_hours'], 5.75)
            self.assertAlmostEqual(root['review10_package_actual_plus_remaining_caps'], pack['actual_card_hours']+5.75)
            self.assertAlmostEqual(root['unallocated_gpu_hours'], 10-prior['actual_card_hours']-5.75)
            self.assertEqual(root['jobs'][-1]['actual_card_hours'], .1)

    def test_unsafe_deferred_structure_freezes_available_funds_without_resetting_fees(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            package = home/'recovery'
            package.mkdir()
            self.stage_fixture(package)
            stage = package/'teacher'
            job = parent()
            saved_groups = groups()
            saved_groups['teacher'].update(jobs=[job], submitted_job_ids=['701'])
            (stage/'BUDGET.json').write_text(json.dumps(saved_groups['teacher']))
            (stage/'SUBMISSIONS.json').write_text(json.dumps(dict(jobs=[job])))
            prior, pack = control.aggregate_budget(baseline(), authorization(), saved_groups)
            (home/'BUDGET.json').write_text(json.dumps(prior))
            (package/'BUDGET.json').write_text(json.dumps(pack))
            def unsafe_saved_observation(name):
                bad = copy.deepcopy(saved_groups[name])
                bad['jobs'][0]['cpu_cores'] = 7
                bad.update(accounting_complete=False, accounting_unknown_job_ids=['701'],
                           collection_risks=['invalid_saved_allocation'])
                (stage/'BUDGET.json').write_text(json.dumps(bad))
            with patch.object(control, 'HOME', home), patch.object(control, 'PACKAGE', package), \
                 patch.object(control, 'validate_frozen_history'), \
                 patch.object(control, 'recover_stage', side_effect=unsafe_saved_observation):
                with self.assertRaises(ValueError):
                    control.main(['--collect', '--stage', 'teacher'])
            for path, old, cap in ((home/'BUDGET.json', prior, 10), (package/'BUDGET.json', pack, 6.5)):
                saved = json.loads(path.read_text())
                self.assertEqual(saved['actual_card_hours'], old['actual_card_hours'])
                self.assertIs(saved.get('spending_frozen'), True)
                self.assertAlmostEqual(saved['remaining_caps'], cap-saved['actual_card_hours'])
                self.assertAlmostEqual(saved['actual_plus_remaining_caps'], cap)
                self.assertEqual(saved['unallocated_gpu_hours'], 0)
                self.assertEqual(saved['accounting_unknown_job_ids'], ['701'])
                if cap == 10:
                    self.assertAlmostEqual(saved['review10_reservation_card_hours'], 6.5-pack['actual_card_hours'])
                    self.assertAlmostEqual(saved['review10_package_actual_plus_remaining_caps'], 6.5)


class V22BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(control)

    def test_six_history_and_two_package_fees_are_exposed_once(self):
        root, package = control.aggregate_budget(baseline(), authorization(), groups())
        self.assertEqual(set(control.FROZEN_SECONDS), set(SECONDS))
        self.assertAlmostEqual(package['frozen652_actual_card_hours'],507/3600)
        self.assertAlmostEqual(package['frozen656_actual_card_hours'],153/3600)
        self.assertAlmostEqual(package['historical_package_actual_card_hours'],660/3600)
        self.assertEqual(package['authorized_card_hours'],6.5)
        self.assertEqual(package['new_actual_card_hours'],0)
        self.assertEqual(package['jobs'],[])
        self.assertAlmostEqual(package['actual_plus_remaining_caps'],6.033333333333333)
        self.assertAlmostEqual(root['actual_plus_remaining_caps'],6.075555555555555)
        self.assertEqual({j['slurm_job_id'] for j in root['jobs']},set(SECONDS))

    def test_omitted656_changed_fee_or_six_cap_is_rejected(self):
        base, auth = baseline(), authorization()
        for mutate in (lambda b,a:b['jobs'].pop(),
                       lambda b,a:b['jobs'][-1].update(actual_card_hours=0),
                       lambda b,a:a.update(package_total_cap_including652_656_card_hours=6),
                       lambda b,a:a.update(historical_package_actual_card_hours=507/3600),
                       lambda b,a:a.update(historical_root_actual_card_hours=659/3600)):
            b,a=copy.deepcopy((base,auth));mutate(b,a)
            with self.assertRaises(ValueError): control.aggregate_budget(b,a,groups())

    def test_both_old_teachers_cannot_be_new_parents(self):
        for sid in ('652','656'):
            g=groups();g['teacher'].update(jobs=[parent(sid=sid)],submitted_job_ids=[sid])
            with self.assertRaises(ValueError): control.aggregate_budget(baseline(),authorization(),g)

    def test_all328_history_files_and_manifest_bytes_are_checked(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);package=root/'package';package.mkdir()
            files={}
            for i in range(328):
                p=root/f'old/{i}.json';p.parent.mkdir(exist_ok=True);p.write_text(str(i))
                files[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
            manifest=package/'HISTORY_IMMUTABLE_SHA256.json'
            manifest.write_text(json.dumps(dict(count=328,files=files)))
            frozen=package/'BASELINE_BUDGET.json';frozen.write_text(json.dumps(baseline()))
            with patch.object(control,'ROOT',root),patch.object(control,'PACKAGE',package), \
                 patch.object(control,'HISTORY_SHA256',hashlib.sha256(manifest.read_bytes()).hexdigest()), \
                 patch.object(control,'BASELINE_SHA256',hashlib.sha256(frozen.read_bytes()).hexdigest()):
                control.validate_frozen_history()
                (root/'old/327.json').write_text('tampered')
                with self.assertRaises(ValueError): control.validate_frozen_history()
                (root/'old/327.json').write_text('327')
                manifest.write_text(manifest.read_text()+' ')
                with self.assertRaises(ValueError): control.validate_frozen_history()

    def test_collect_refuses_old_stage_even_if_output_binding_is_consistent(self):
        job=parent();remote=Path('/remote/training_smoke/claude10/recovery_v21/teacher')
        job.update(remote_output=str(remote/'outputs/teacher_701'),remote_log=str(remote/'slurm-701.out'))
        with self.assertRaises(ValueError):
            control._manifest_job('teacher',dict(jobs=[job]),dict(jobs=[job],remote_smoke_root=str(remote)),
                                  dict(authorized_card_hours=2,jobs=[job]))


    def test_historical_raw_allocation_proof_must_keep_one_GPU_six_CPU_90GiB(self):
        for tres in ('cpu=12,mem=90G,gres/gpu=1','cpu=6,mem=80G,gres/gpu=1',
                     'cpu=6,mem=90G,gres/gpu=2'):
            b=baseline();b['jobs'][-1]['sacct']['alloc_tres']=tres
            with self.assertRaises(ValueError): control.aggregate_budget(b,authorization(),groups())

    def test_withheld_state_requires_an_observed_upstream_failure(self):
        for stage in ('teacher','sft','grpo'):
            g=groups();g[stage]['state']='NOT_RELEASED_UPSTREAM_FAILURE'
            with self.assertRaises(ValueError): control.aggregate_budget(baseline(),authorization(),g)


    def test_zero_second_allocated_parents_still_require_exact_raw_resources(self):
        invalid=['cpu=12,mem=90G,gres/gpu=1,gres/gpu:a100=1',
                 'cpu=6,mem=180G,gres/gpu=1,gres/gpu:a100=1',
                 'cpu=6,mem=90G,gres/gpu=2,gres/gpu:a100=2','']
        for state in ('COMPLETED','FAILED','RUNNING','CANCELLED'):
            for tres in invalid:
                with self.subTest(state=state,tres=tres):
                    g=groups();job=parent(seconds=0,state=state,allocation=1)
                    job['sacct']['alloc_tres']=tres
                    g['teacher'].update(jobs=[job],submitted_job_ids=['701'])
                    with self.assertRaises(ValueError):
                        control.aggregate_budget(baseline(),authorization(),g)

    def test_zero_allocation_exception_requires_empty_raw_proof_and_explicit_pending_state(self):
        for state in ('PENDING','CONFIGURING','CANCELLED'):
            with self.subTest(state=state):
                g=groups();job=parent(seconds=0,state=state,allocation=0)
                g['teacher'].update(jobs=[job],submitted_job_ids=['701'])
                root,_=control.aggregate_budget(baseline(),authorization(),g)
                self.assertTrue(root['accounting_complete'])
                self.assertEqual(root['remaining_caps'],3.85 if state=='CANCELLED' else 5.85)
                for tres in ('cpu=12,mem=180G,gres/gpu=2',None):
                    job['sacct']['alloc_tres']=tres
                    with self.assertRaises(ValueError):
                        control.aggregate_budget(baseline(),authorization(),g)

    def test_zero_second_resource_conflicts_defer_before_reconciliation_and_keep_all_reserve(self):
        invalid=['cpu=12,mem=90G,gres/gpu=1,gres/gpu:a100=1',
                 'cpu=6,mem=180G,gres/gpu=1,gres/gpu:a100=1',
                 'cpu=6,mem=90G,gres/gpu=2,gres/gpu:a100=2']
        for tres in invalid:
            with self.subTest(tres=tres),tempfile.TemporaryDirectory() as tmp:
                home=Path(tmp);package=home/'recovery';package.mkdir()
                RecoveryCLITests.stage_fixture(self,package)
                stage=package/'teacher';job=parent(seconds=0,state='PENDING')
                remote='/remote/training_smoke/claude10/recovery_v22/teacher'
                job.update(remote_output=remote+'/outputs/teacher_701',remote_log=remote+'/slurm-701.out')
                saved=groups();saved['teacher'].update(jobs=[job],submitted_job_ids=['701'])
                (stage/'BUDGET.json').write_text(json.dumps(saved['teacher']))
                (stage/'SUBMISSIONS.json').write_text(json.dumps(dict(jobs=[job])))
                (stage/'JOB_MANIFEST.json').write_text(json.dumps(dict(jobs=[job],remote_smoke_root=remote)))
                prior,pack=control.aggregate_budget(baseline(),authorization(),saved)
                (home/'BUDGET.json').write_text(json.dumps(prior))
                (package/'BUDGET.json').write_text(json.dumps(pack))
                snapshot=dict(ledger={'ledger_status':'RECONCILED'},account=dict(code=0,
                    stdout='701|COMPLETED|0|'+tres+'|0:0|6|1\n'))
                raw=json.dumps(snapshot)
                def observation(*args):
                    (stage/'LATEST.json').write_text(raw);return snapshot
                with patch.object(control,'HOME',home),patch.object(control,'PACKAGE',package), \
                     patch.object(control,'validate_frozen_history'), \
                     patch.object(control.collect,'query',side_effect=observation), \
                     patch.object(control.collect,'reconcile_budget') as reconcile, \
                     patch.object(control.subprocess,'run') as copy_output:
                    with self.assertRaises(ValueError):control.main(['--collect','--stage','teacher'])
                    reconcile.assert_not_called();copy_output.assert_not_called()
                saved_root=json.loads((home/'BUDGET.json').read_text())
                saved_stage=json.loads((stage/'BUDGET.json').read_text())
                self.assertEqual(saved_root['collection_status'],'DEFERRED')
                self.assertFalse(saved_root['accounting_complete'])
                self.assertEqual(saved_root['accounting_unknown_job_ids'],['701'])
                self.assertEqual(saved_root['remaining_caps'],5.85)
                self.assertEqual(saved_root['actual_card_hours'],prior['actual_card_hours'])
                self.assertTrue(saved_stage['collection_risks'])
                self.assertEqual((stage/'LATEST.json').read_text(),raw)

    def test_baseline_cannot_replace_frozen_positive_seconds_with_zero(self):
        for index in range(6):
            b=baseline();b['jobs'][index]['sacct']['elapsed_seconds']=0
            with self.assertRaises(ValueError):control.validate_baseline(b)


class ZeroAllocationHelperIntegrationTests(unittest.TestCase):
    """Execute the real local helper in-process; every external command is intercepted."""
    def fixture(self,state,tres=''):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        home=Path(tmp.name);package=home/'recovery';package.mkdir()
        RecoveryCLITests.stage_fixture(self,package)
        stage=package/'teacher';job=parent(seconds=0,state='PENDING',allocation=0)
        remote='/remote/training_smoke/claude10/recovery_v22/teacher'
        job.update(remote_output=remote+'/outputs/teacher_701',remote_log=remote+'/slurm-701.out')
        saved=groups();saved['teacher'].update(jobs=[job],submitted_job_ids=['701'])
        (stage/'BUDGET.json').write_text(json.dumps(saved['teacher']))
        (stage/'SUBMISSIONS.json').write_text(json.dumps(dict(jobs=[job])))
        (stage/'JOB_MANIFEST.json').write_text(json.dumps(dict(jobs=[job],remote_smoke_root=remote)))
        prior,pack=control.aggregate_budget(baseline(),authorization(),saved)
        (home/'BUDGET.json').write_text(json.dumps(prior))
        (package/'BUDGET.json').write_text(json.dumps(pack))
        snapshot=dict(ledger={'ledger_status':'RECONCILED'},account=dict(code=0,
            stdout='701|'+state+'|0|'+tres+'|0:0|0|0\n'))
        def query(*args):
            (stage/'LATEST.json').write_text(json.dumps(snapshot));return snapshot
        return home,package,stage,query

    def local_command(self,args,**kwargs):
        import contextlib
        import io
        import subprocess
        import smoke_collect
        expected=[sys.executable,str(control.collect.ROOT/'scripts/smoke_collect.py')]
        if args[:2]==expected:
            with patch.object(sys,'argv',args[1:]),contextlib.redirect_stdout(io.StringIO()) as output:
                smoke_collect.main()
            return subprocess.CompletedProcess(args,0,output.getvalue(),'')
        if args[:2]==['scp','-r']:
            return subprocess.CompletedProcess(args,0,'','')
        self.fail('All unexpected external commands are forbidden in offline tests: '+repr(args))

    def test_verified_zero_allocation_survives_real_local_helper_for_all_three_allowed_states(self):
        for state in ('PENDING','CONFIGURING','CANCELLED'):
            with self.subTest(state=state):
                home,package,stage,query=self.fixture(state)
                with patch.object(control,'HOME',home),patch.object(control,'PACKAGE',package), \
                     patch.object(control,'validate_frozen_history'), \
                     patch.object(control.collect,'query',side_effect=query), \
                     patch.object(control.subprocess,'run',side_effect=self.local_command) as commands:
                    try:control.main(['--collect','--stage','teacher'])
                    except ValueError as error:self.fail('Verified zero allocation must survive helper: '+str(error))
                saved=json.loads((stage/'BUDGET.json').read_text())
                job=saved['jobs'][0]
                self.assertEqual(job['allocation_gpu_count'],0)
                self.assertEqual(job['actual_card_hours'],0)
                self.assertEqual(job['sacct']['alloc_tres'],'')
                self.assertEqual(job['sacct']['elapsed_seconds'],0)
                self.assertEqual(job['state'],state)
                root=json.loads((home/'BUDGET.json').read_text())
                self.assertTrue(root['accounting_complete'])
                self.assertEqual(root['actual_card_hours'],812/3600)
                self.assertEqual(root['remaining_caps'],3.85 if state=='CANCELLED' else 5.85)
                self.assertEqual(sum(call.args[0][0]==sys.executable for call in commands.call_args_list),1)
                self.assertEqual(sum(call.args[0][0]=='scp' for call in commands.call_args_list),
                                 2 if state=='CANCELLED' else 0)

    def test_conflicting_zero_second_raw_resources_never_reach_real_helper_or_copy(self):
        for tres in ('cpu=12,mem=90G,gres/gpu=1','cpu=6,mem=180G,gres/gpu=1',
                     'cpu=6,mem=90G,gres/gpu=2'):
            with self.subTest(tres=tres):
                home,package,stage,query=self.fixture('COMPLETED',tres)
                with patch.object(control,'HOME',home),patch.object(control,'PACKAGE',package), \
                     patch.object(control,'validate_frozen_history'), \
                     patch.object(control.collect,'query',side_effect=query), \
                     patch.object(control.subprocess,'run',side_effect=self.local_command) as commands:
                    with self.assertRaises(ValueError):control.main(['--collect','--stage','teacher'])
                    commands.assert_not_called()
                root=json.loads((home/'BUDGET.json').read_text())
                self.assertFalse(root['accounting_complete'])
                self.assertEqual(root['collection_status'],'DEFERRED')
                self.assertEqual(root['remaining_caps'],5.85)
                self.assertEqual(root['actual_card_hours'],812/3600)


class StageHeaderPublicationTests(unittest.TestCase):
    def fixture(self,job=None,complete=True,unknown=None,pending=False):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        home=Path(tmp.name);package=home/'recovery';package.mkdir()
        RecoveryCLITests.stage_fixture(self,package)
        group=groups()['teacher'];amount=job['actual_card_hours'] if job else 0
        group.update(jobs=[job] if job else [],actual_card_hours=amount,remaining_caps=2,
            actual_plus_remaining_caps=amount,state='SUBMITTED',accounting_complete=complete,
            accounting_unknown_job_ids=[] if unknown is None else unknown)
        ledger=dict(jobs=[job] if job else [])
        if pending:ledger['pending_submission']={'claim_id':'a'*32}
        (package/'teacher/BUDGET.json').write_text(json.dumps(group))
        (package/'teacher/SUBMISSIONS.json').write_text(json.dumps(ledger))
        return home,package

    def publish(self,home,package):
        with patch.object(control,'HOME',home),patch.object(control,'PACKAGE',package), \
             patch.object(control,'validate_frozen_history'), \
             patch.object(control.collect,'query') as query, \
             patch.object(control.subprocess,'run') as command:
            control.main(['--publish'])
            query.assert_not_called();command.assert_not_called()
        return json.loads((package/'teacher/BUDGET.json').read_text())

    def test_publish_closes_stale_submitted_header_using_saved_verified_terminal_parent_only(self):
        job=parent(seconds=1312,state='COMPLETED');home,package=self.fixture(job)
        ledger=(package/'teacher/SUBMISSIONS.json').read_bytes()
        saved=self.publish(home,package)
        self.assertEqual(saved['remaining_caps'],0)
        self.assertEqual(saved['state'],'COMPLETED')
        self.assertAlmostEqual(saved['actual_plus_remaining_caps'],1312/3600)
        self.assertEqual(saved['actual_card_hours'],job['actual_card_hours'])
        self.assertEqual(saved['jobs'],[job])
        self.assertEqual((package/'teacher/SUBMISSIONS.json').read_bytes(),ledger)
        root=json.loads((home/'BUDGET.json').read_text())
        pack=json.loads((package/'BUDGET.json').read_text())
        self.assertAlmostEqual(root['actual_card_hours'],.59)
        self.assertEqual(root['remaining_caps'],3.85)
        self.assertEqual(pack['stage_states']['teacher'],'COMPLETED')

    def test_collect_real_legacy_helper_also_publishes_consistent_terminal_stage_header(self):
        job=parent(seconds=0,state='PENDING',allocation=0)
        remote='/remote/training_smoke/claude10/recovery_v22/teacher'
        job.update(remote_output=remote+'/outputs/teacher_701',remote_log=remote+'/slurm-701.out')
        home,package=self.fixture(job)
        (package/'teacher/JOB_MANIFEST.json').write_text(json.dumps(dict(jobs=[job],remote_smoke_root=remote)))
        snapshot=dict(ledger={'ledger_status':'RECONCILED'},account=dict(code=0,
            stdout='701|COMPLETED|1312|cpu=6,mem=90G,gres/gpu=1,gres/gpu:a100=1|0:0|6|1\n'))
        with patch.object(control,'HOME',home),patch.object(control,'PACKAGE',package), \
             patch.object(control,'validate_frozen_history'), \
             patch.object(control.collect,'query',return_value=snapshot), \
             patch.object(control.subprocess,'run',side_effect=lambda *args,**kwargs:
                 ZeroAllocationHelperIntegrationTests.local_command(self,*args,**kwargs)) as commands:
            control.main(['--collect','--stage','teacher'])
        saved=json.loads((package/'teacher/BUDGET.json').read_text())
        self.assertEqual(saved['remaining_caps'],0)
        self.assertEqual(saved['state'],'COMPLETED')
        self.assertAlmostEqual(saved['actual_plus_remaining_caps'],1312/3600)
        self.assertAlmostEqual(saved['actual_card_hours'],1312/3600)
        self.assertEqual(saved['jobs'][0]['sacct']['elapsed_seconds'],1312)
        self.assertEqual(sum(call.args[0][0]==sys.executable for call in commands.call_args_list),1)
        self.assertEqual(sum(call.args[0][0]=='scp' for call in commands.call_args_list),2)

    def test_active_and_unknown_parent_headers_keep_unused_cap_instead_of_releasing(self):
        cases=[('RUNNING',True,[], 'RUNNING'),('COMPLETED',False,['701'],'ACCOUNTING_UNKNOWN'),
               ('COMPLETED',False,[],'ACCOUNTING_UNKNOWN'),('COMPLETED',True,['701'],'ACCOUNTING_UNKNOWN')]
        for state,complete,unknown,expected in cases:
            with self.subTest(state=state,complete=complete,unknown=unknown):
                job=parent(seconds=360,state=state);home,package=self.fixture(job,complete,unknown)
                saved=self.publish(home,package)
                self.assertAlmostEqual(saved['remaining_caps'],1.9)
                self.assertAlmostEqual(saved['actual_plus_remaining_caps'],2)
                self.assertEqual(saved['state'],expected)
                self.assertEqual(saved['accounting_complete'],expected!='ACCOUNTING_UNKNOWN')
                self.assertEqual(saved['jobs'],[job])

    def test_pending_intent_keeps_entire_stage_cap_and_cannot_appear_terminal(self):
        home,package=self.fixture(pending=True)
        saved=self.publish(home,package)
        self.assertEqual(saved['remaining_caps'],2)
        self.assertEqual(saved['actual_plus_remaining_caps'],2)
        self.assertEqual(saved['state'],'SUBMISSION_UNCERTAIN')
        self.assertFalse(saved['accounting_complete'])
        self.assertEqual(saved['jobs'],[])

    def test_withheld_empty_headers_release_only_after_existing_upstream_failure_validation(self):
        home,package=self.fixture(parent(state='FAILED'))
        for stage in ('sft','grpo'):
            path=package/stage/'BUDGET.json';group=json.loads(path.read_text())
            group.update(state='NOT_RELEASED_UPSTREAM_FAILURE',remaining_caps=CAPS[stage],
                actual_plus_remaining_caps=CAPS[stage]);path.write_text(json.dumps(group))
        self.publish(home,package)
        for stage in ('sft','grpo'):
            saved=json.loads((package/stage/'BUDGET.json').read_text())
            self.assertEqual(saved['remaining_caps'],0)
            self.assertEqual(saved['actual_plus_remaining_caps'],0)
            self.assertEqual(saved['state'],'NOT_RELEASED_UPSTREAM_FAILURE')
        home2,package2=self.fixture()
        path=package2/'sft/BUDGET.json';group=json.loads(path.read_text())
        group['state']='NOT_RELEASED_UPSTREAM_FAILURE';path.write_text(json.dumps(group))
        before={s:(package2/s/'BUDGET.json').read_bytes() for s in CAPS}
        with self.assertRaises(ValueError):self.publish(home2,package2)
        self.assertEqual(before,{s:(package2/s/'BUDGET.json').read_bytes() for s in CAPS})

    def test_invalid_saved_allocation_cannot_trigger_any_header_normalization(self):
        job=parent(seconds=1312);job['sacct']['alloc_tres']='cpu=6,mem=180G,gres/gpu=1'
        home,package=self.fixture(job)
        before={s:(package/s/'BUDGET.json').read_bytes() for s in CAPS}
        with self.assertRaises(ValueError):self.publish(home,package)
        self.assertEqual(before,{s:(package/s/'BUDGET.json').read_bytes() for s in CAPS})


if __name__ == '__main__':
    unittest.main()
