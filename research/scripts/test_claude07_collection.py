"""Offline regressions for collection, separated billing, and review07 gates."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class CollectionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.c = importlib.import_module('collect_claude07')
        except ImportError:
            self.fail('Claude07 collector is missing')

    def test_gpu_parent_elapsed_and_cpu_parent_time_are_billed_once(self):
        gpu = dict(name='g', gpus=1, cores=6, cap=3)
        parent = dict(JobIDRaw='12', State='FAILED', ExitCode='124:0',
                      ElapsedRaw='3600', CPUTimeRAW='21600', AllocCPUS='6', AllocTRES='cpu=6,gres/gpu=1')
        self.assertEqual(self.c.account_job(gpu, parent)['actual'], 1)
        cpu = dict(name='c', gpus=0, cores=32, cap=32)
        parent.update(AllocCPUS='32', CPUTimeRAW='115200', AllocTRES='cpu=32')
        self.assertEqual(self.c.account_job(cpu, parent)['actual'], 32)

    def test_allocation_mismatch_is_pool_risk_not_assertion(self):
        row = dict(State='RUNNING', ElapsedRaw='60', CPUTimeRAW='720', AllocCPUS='12', AllocTRES='cpu=12,gres/gpu=2')
        r = self.c.account_job(dict(name='g', gpus=1, cores=6), row)
        self.assertAlmostEqual(r['actual'], 2/60)
        self.assertTrue(r['allocation_anomalies'])

    def test_zero_elapsed_pending_requires_no_allocation(self):
        r = self.c.account_job(dict(name='g', gpus=1, cores=6), dict(State='PENDING', ElapsedRaw='0'))
        self.assertEqual(r['actual'], 0)
        self.assertEqual(r['allocation_anomalies'], [])

    def test_stop_marker_with_bad_json_remains_present_and_preserves_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'STOP.json'
            path.write_text('{broken\n')
            value = self.c.read_marker(path)
            self.assertTrue(value['present'])
            self.assertEqual(value['raw_text'], '{broken\n')
            self.assertIn('parse_error', value)

    def test_failure_only_cancels_scientific_descendants(self):
        jobs = [dict(name=n, cap=1, depends_on=deps, resource_afterany=lane)
                for n, deps, lane in [('bad', [], []), ('independent', [], ['bad']),
                                      ('child', ['bad'], []), ('grandchild', ['child'], [])]]
        rows = [dict(name=n, state=s, actual=0, usable=False, allocation_anomalies=[])
                for n, s in [('bad', 'FAILED'), ('independent', 'PENDING'),
                             ('child', 'PENDING'), ('grandchild', 'RUNNING')]]
        p = self.c.guard_plan(jobs, rows, dict(safe=True, reasons=[]), None)
        self.assertEqual(p['cancel_names'], ['child'])
        self.assertFalse(p['pool_stop_required'])

    def test_pool_risk_cancels_pending_only_and_preserves_running(self):
        jobs = [dict(name=n, cap=1, depends_on=[]) for n in ['a', 'b', 'c']]
        rows = [dict(name=n, state=s, actual=0, usable=False, allocation_anomalies=[])
                for n, s in [('a', 'FAILED'), ('b', 'PENDING'), ('c', 'RUNNING')]]
        p = self.c.guard_plan(jobs, rows, dict(safe=False, reasons=['round_cap_exceeded']), None)
        self.assertEqual(p['cancel_names'], ['b'])
        self.assertTrue(p['pool_stop_required'])

    def test_runtime_failed_status_blocks_dependents_before_sacct_catches_up(self):
        jobs = [dict(name='bad', depends_on=[]), dict(name='child', depends_on=['bad'])]
        rows = [dict(name='bad', state='RUNNING', actual=0, usable=False,
                     allocation_anomalies=[], task_status={'present': True, 'data': {'state': 'FAILED'}}),
                dict(name='child', state='PENDING', actual=0, usable=False, allocation_anomalies=[])]
        p = self.c.guard_plan(jobs, rows, dict(safe=True, reasons=[]), None)
        self.assertEqual(p['cancel_names'], ['child'])
        self.assertFalse(p['pool_stop_required'])

    def test_failed_terminal_xml_still_retains_full_frame_for_diagnostics(self):
        import shutil
        root = Path(__file__).resolve().parents[1]
        old = root/'calibration_gpu/claude06_outputs/c06_L00_relax_fromideal'
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp); directory = home/'outputs/continue'; directory.mkdir(parents=True)
            for name in ['vasprun.xml', 'OUTCAR']:
                shutil.copyfile(old/name, directory/name)
            job = dict(name='continue', kind='relax', n_atoms=316, gpus=1, cores=6)
            account = dict(State='FAILED', ExitCode='124:0', ElapsedRaw='14282',
                           AllocCPUS='6', CPUTimeRAW='85692', AllocTRES='cpu=6,gres/gpu=1')
            data = dict(parents={'596': account}, steps=[], files={})
            result = self.c.analyze_one(root, home, job, {'slurm_job_id': '596'}, data)
            self.assertFalse(result['usable'])
            self.assertEqual(result['last_complete_frame']['completed_calculations'], 108)
            self.assertEqual(len(result['final_geometry']['scaled_positions']), 316)
            self.assertEqual(len(result['forces_eV_A']), 316)
            self.assertTrue(result['SCF_loop_real_seconds'])

    def test_bad_task_status_is_preserved_without_crashing_guard(self):
        row = dict(name='a', state='PENDING', actual=0, usable=False,
                   allocation_anomalies=[], task_status={'present': True, 'data': ['bad schema'], 'raw_text': '["bad schema"]'})
        plan = self.c.guard_plan([dict(name='a', depends_on=[])], [row], dict(safe=True, reasons=[]), None)
        self.assertTrue(plan['task_status_errors'])
        self.assertEqual(plan['cancel_names'], [])

    def test_guard_rechecks_state_and_uses_pending_filter(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); remote = root/'remote'; remote.mkdir()
            home = root/'home'; (home/'evidence').mkdir(parents=True)
            (remote/'CLAUDE07_POOL_STOP.json').write_text('{broken raw')
            binary = root/'bin'; binary.mkdir()
            (binary/'squeue').write_text('#!/bin/sh\ncase "$*" in *"-j 123"*) exit 0;; *) printf "123|PENDING\\n456|RUNNING\\n";; esac\n')
            (binary/'scancel').write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> '+str(root/'cancelled')+'\n')
            for p in binary.iterdir():p.chmod(0o755)
            settings = dict(host='fake', remote=str(remote))
            jobs = [dict(name='a', remote_directory=str(remote/'a'))]
            data = dict(ledger={'jobs': [dict(name='a', slurm_job_id='123')]})
            plan = dict(cancel_names=['a'], blocked_descendants=[], failed_tasks=[],
                        pool_stop_required=True, pool_stop_reasons=[], existing_stop={'present': True, 'raw_text': '{broken raw'})
            def fake_ssh(host, command, payload, timeout):
                import os
                env = dict(os.environ, PATH=str(binary)+':'+os.environ['PATH'])
                return subprocess.run(['python3', '-'], input=payload, capture_output=True, env=env, check=True)
            with patch.object(self.c, 'ssh', fake_ssh):
                result = self.c.apply_guard(home, settings, jobs, data, plan)
            self.assertEqual(result['cancelled_only_pending_own_ids'], ['123'])
            self.assertEqual((root/'cancelled').read_text().strip(), '--state=PENDING 123')
            self.assertEqual((remote/'CLAUDE07_POOL_STOP.json').read_text(), '{broken raw')

    def test_guard_does_not_mark_running_task_blocked_after_queue_race(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); remote = root/'remote'; (remote/'child').mkdir(parents=True)
            home = root/'home'; (home/'evidence').mkdir(parents=True)
            binary = root/'bin'; binary.mkdir()
            (binary/'squeue').write_text('#!/bin/sh\ncase "$*" in *"-j 123"*) printf "RUNNING\\n";; *) printf "123|PENDING\\n";; esac\n')
            (binary/'scancel').write_text('#!/bin/sh\nexit 0\n')
            for p in binary.iterdir():p.chmod(0o755)
            plan = dict(cancel_names=['child'], blocked_descendants=['child'], failed_tasks=['bad'],
                        pool_stop_required=False, pool_stop_reasons=[], existing_stop=None)
            def fake_ssh(host, command, payload, timeout):
                import os
                env = dict(os.environ, PATH=str(binary)+':'+os.environ['PATH'])
                return subprocess.run(['python3', '-'], input=payload, capture_output=True, env=env, check=True)
            with patch.object(self.c, 'ssh', fake_ssh):
                result = self.c.apply_guard(home, dict(host='fake', remote=str(remote)),
                    [dict(name='child', remote_directory=str(remote/'child'))],
                    dict(ledger={'jobs': [dict(name='child', slurm_job_id='123')]}), plan)
            self.assertFalse((remote/'child/TASK_STATUS.json').exists())
            self.assertIn('123', result['started_before_cancellation'])


class ReportTests(unittest.TestCase):
    def setUp(self):
        try:
            self.r = importlib.import_module('render_claude07_report')
        except ImportError:
            self.fail('Claude07 report is missing')

    def row(self, config, energy, same=True):
        return dict(config=config, energy_eV={'e_0_energy': energy}, usable=True,
                    actual=.1, convergence={'n_ionic_outputs': 2},
                    occupancy={'same_MACE_DFT_occupation_class': same},
                    input_geometry_matches_mace=True)

    def test_g2_common_energy_shift_is_removed_before_span_gate(self):
        base = {'B10': -10, 'B11': -9.9, 'B15': -9.8, 'B00': -8}
        rows = [self.row(k, v-5+i*.01) for i, (k, v) in enumerate(base.items())]
        result = self.r.g2_gate(rows, base)
        self.assertTrue(result['supports_SP_plus_topk_relaxation'])
        self.assertAlmostEqual(result['relative_correction_span_eV'], .03)
        self.assertFalse(result['top3_order_changed'])
        self.assertAlmostEqual(result['per_configuration'][0]['relative_relax_minus_SP_eV'], 0)

    def test_g2_occupancy_change_rejects_sp_even_for_small_energy_span(self):
        base = {k: i for i, k in enumerate(['B10', 'B11', 'B15', 'B00'])}
        rows = [self.row(k, v, k != 'B15') for k, v in base.items()]
        self.assertFalse(self.r.g2_gate(rows, base)['supports_SP_plus_topk_relaxation'])

    def test_g2_exact_occupation_change_rejects_even_if_canonical_equivalent(self):
        base = {k: i for i, k in enumerate(['B10', 'B11', 'B15', 'B00'])}
        rows = [self.row(k, v) for k, v in base.items()]
        rows[-1]['occupancy']['same_exact_occupation'] = False
        self.assertFalse(self.r.g2_gate(rows, base)['supports_SP_plus_topk_relaxation'])

    def test_g2_at_strict_threshold_is_not_released(self):
        base = {k: 0 for k in ['B10', 'B11', 'B15', 'B00']}
        rows = [self.row(k, .1 if k == 'B00' else 0) for k in base]
        self.assertFalse(self.r.g2_gate(rows, base)['supports_SP_plus_topk_relaxation'])

    def test_g2_missing_geometry_or_pending_never_becomes_positive_label(self):
        rows = [self.row('B10', -1)]
        self.assertIsNone(self.r.g2_gate(rows, {'B10': -1})['supports_SP_plus_topk_relaxation'])

    def test_g3_pairwise_corrections_use_gamma_same_geometry(self):
        base = {'B10': -10, 'B07': -9, 'B00': -8}
        rows = [self.row(k, v-2+i*.02) for i, (k, v) in enumerate(base.items())]
        r = self.r.g3_gate(rows, base)
        self.assertTrue(r['passed'])
        self.assertAlmostEqual(r['max_pairwise_relative_change_eV'], .04)
        rows[-1]['input_geometry_matches_mace'] = False
        self.assertIsNone(self.r.g3_gate(rows, base)['passed'])

    def test_s00_offset_is_explicit_inference_and_new_s08_updates_three_points(self):
        old = dict(LREAL_paired_check={'rows': [{'Auto_minus_False_eV': -.02}]},
                   numerical_convergence=[dict(variant='k333',
                       per_config_high_minus_base_eV={'S00': -.023, 'S04': -.002})])
        result = self.r.corrected_convergence(old, -.004, protocol_verified=True)[0]
        self.assertEqual(result['n_configs'], 3)
        self.assertAlmostEqual(result['per_config_corrected_high_minus_base_eV']['S00'], -.003)
        self.assertEqual(result['S00_evidence_kind'], 'common_offset_inference')
        self.assertFalse(result['S00_direct_same_protocol_measurement'])
        self.assertTrue(result['within_0_02_eV'])

    def test_real_high_precision_protocol_matches_recorded_runtime_hashes(self):
        root = Path(__file__).resolve().parents[1]
        protocol = self.r.verify_protocol(root)
        self.assertTrue(protocol['verified'])
        self.assertTrue(all(row['runtime_INCAR_hash_matches'] for row in protocol['rows']))

    def test_report_shows_review08_reservation_without_rewriting_submitted_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke = root/'training_smoke'; smoke.mkdir()
            budget = {'actual_card_hours': 0, 'actual_plus_remaining_caps': 1.25,
                      'authorized_card_hours': 10}
            (smoke/'BUDGET.json').write_text(json.dumps(budget))
            authorization = {'conditional_grpo': {'state': 'AUTHORIZED_CONDITIONAL_NOT_SUBMITTED',
                'cap_card_hours': 2, 'slurm_job_id': None},
                'submitted_plus_conditional_hard_ceiling_card_hours': 3.25,
                'unassigned_after_conditional_reservation_card_hours': 6.75}
            (smoke/'REVIEW08_AUTHORIZATION.json').write_text(json.dumps(authorization))
            result = self.r.render(root)
            report = (root/'CLAUDE07_RESULTS.md').read_text()
            self.assertEqual(result['review08_smoke_authorization'], authorization)
            self.assertIn('6.75', report)
            self.assertNotIn('其余8.75未分配', report)
            self.assertEqual(json.loads((smoke/'BUDGET.json').read_text()), budget)

    def test_withheld_grpo_uses_current_parent_cost_and_pending_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke = root/'training_smoke'; smoke.mkdir()
            budget = {'actual_card_hours': .0025, 'actual_plus_remaining_caps': .5025,
                      'authorized_card_hours': 10}
            (smoke/'BUDGET.json').write_text(json.dumps(budget))
            authorization = {'conditional_grpo': {
                'state': 'NOT_RELEASED_PREREQUISITE_MACE635_FAILED',
                'cap_card_hours': 2, 'slurm_job_id': None,
                'not_released_reason': '635 failed before its first E/F call'},
                'additional_conditional_reservation_card_hours': 0,
                'submitted_plus_conditional_hard_ceiling_card_hours': 3.25}
            (smoke/'REVIEW08_AUTHORIZATION.json').write_text(json.dumps(authorization))
            self.r.render(root)
            report = (root/'CLAUDE07_RESULTS.md').read_text()
            self.assertIn('0.502500', report)
            self.assertIn('9.497500', report)
            self.assertIn('当前不释放GRPO', report)
            self.assertEqual(json.loads((smoke/'BUDGET.json').read_text()), budget)

    def test_mapping_accepts_physical_initial_sites_and_complete_vacancy_sites(self):
        import numpy as np
        from ase.io import write
        parent = self.r.reference(2)
        initial = parent.copy()
        initial.numbers[0] = 39
        vacancy = int(np.flatnonzero(parent.numbers == 8)[0])
        initial.numbers[vacancy] = 0
        initial.set_array('site_id', np.arange(len(initial)))
        physical = initial[initial.numbers != 0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root/'mace.extxyz', physical)
            for source in [initial, physical]:
                write(root/'initial.extxyz', source)
                result = self.r.mapped_comparison(root, dict(parent_geometry='initial.extxyz',
                    mace_geometry='mace.extxyz'), physical)
                self.assertTrue(result['same_exact_occupation'])
                self.assertEqual(result['DFT']['vacancy_site_ids'], [vacancy])
                self.assertFalse(result['DFT']['occupation_class_changed'])

    def test_physical_initial_mapping_rejects_unverified_site_metadata(self):
        import numpy as np
        from ase.io import write
        parent = self.r.reference(2)
        initial = parent.copy()
        initial.set_array('site_id', np.arange(len(initial)))
        vacancy = int(np.flatnonzero(parent.numbers == 8)[0])
        initial = initial[np.arange(len(initial)) != vacancy]
        bad_cases = {}
        a = initial.copy(); del a.arrays['site_id']; bad_cases['missing_ids'] = a
        a = initial.copy(); a.arrays['site_id'][1] = a.arrays['site_id'][0]; bad_cases['duplicate_ids'] = a
        a = initial.copy(); a.arrays['site_id'][0] = len(parent); bad_cases['out_of_range_ids'] = a
        a = initial.copy(); ids = a.arrays.pop('site_id'); a.set_array('site_id', ids.astype(float)+.1); bad_cases['noninteger_ids'] = a
        a = initial.copy(); a.positions[0, 0] += .1; bad_cases['off_lattice'] = a
        a = initial.copy(); a.cell[0, 0] += .1; bad_cases['wrong_cell'] = a
        a = initial.copy(); a.numbers[0] = 8; bad_cases['wrong_sublattice'] = a
        a = initial.copy(); a.pbc[0] = False; bad_cases['nonperiodic'] = a
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root/'mace.extxyz', initial)
            for label, source in bad_cases.items():
                with self.subTest(label=label):
                    write(root/'initial.extxyz', source)
                    with self.assertRaises(ValueError):
                        self.r.mapped_comparison(root, dict(parent_geometry='initial.extxyz',
                            mace_geometry='mace.extxyz'), initial)


class SyncTests(unittest.TestCase):
    def test_single_gpu_user_limit_survives_status_and_report_refreshes(self):
        updater = importlib.import_module('update_claude07_status')
        reporter = importlib.import_module('render_claude07_report')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'calibration').mkdir()
            (root/'calibration/REVIEW_STATUS.json').write_text(json.dumps({
                'max_concurrent_gpu': 1, 'gpu_platform_max_active_GPU_allocations': 2}))
            for _ in range(2):
                updater.update(root)
                current = json.loads((root/'calibration/claude07/STATUS.json').read_text())
                self.assertEqual(current['max_concurrent_gpu'], 1)
                result = reporter.render(root)
                self.assertEqual(result['max_concurrent_gpu'], 1)
                self.assertIn('单卡串行', (root/'CLAUDE07_RESULTS.md').read_text())

    def test_status_refresh_preserves_review08_conditional_scope(self):
        updater = importlib.import_module('update_claude07_status')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'training_smoke').mkdir()
            (root/'training_smoke/REVIEW08_AUTHORIZATION.json').write_text(json.dumps({
                'conditional_grpo': {'submissions_allowed': 1, 'cap_card_hours': 2}}))
            updater.update(root)
            status = json.loads((root/'calibration/REVIEW_STATUS.json').read_text())
            self.assertIn('Review08', status['further_submissions_scope'])
            self.assertFalse(status['further_submissions_allowed'])
            self.assertFalse(status['formal_training_allowed'])

    def test_sync_includes_review08_design_and_conditional_authorization(self):
        sync = importlib.import_module('sync_claude07_control')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root/'calibration/claude07'; home.mkdir(parents=True)
            smoke = root/'training_smoke'; smoke.mkdir()
            expected = ['REVIEW_CLAUDE_08_RL_arm.md', 'CLAUDE08_EXECUTION.md',
                        'CLAUDE08_FOLLOWUP.md', 'BUDGET_REPLAN.md']
            for name in expected:
                (root/name).write_text('Review08 summary')
            (smoke/'REVIEW08_AUTHORIZATION.json').write_text('{}')
            (smoke/'SCHEDULE_AMENDMENTS.json').write_text('{}')
            names = {p.name for p in sync.control_files(root, home)}
            self.assertTrue(set(expected+['REVIEW08_AUTHORIZATION.json', 'SCHEDULE_AMENDMENTS.json']) <= names)

    def test_sync_bundle_never_contains_runtime_or_old_stop_files(self):
        try:
            sync = importlib.import_module('sync_claude07_control')
        except ImportError:
            self.fail('Claude07 control synchronization is missing')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); home = root/'calibration/claude07'; home.mkdir(parents=True)
            for name in ['BUDGET.json', 'ANALYSIS.json', 'CLAUDE07_POOL_STOP.json', 'TASK_STATUS.json']:
                (home/name).write_text('{}')
            names = [p.name for p in sync.control_files(root, home)]
            self.assertIn('BUDGET.json', names)
            self.assertNotIn('CLAUDE07_POOL_STOP.json', names)
            self.assertNotIn('TASK_STATUS.json', names)


if __name__ == '__main__':
    unittest.main()
