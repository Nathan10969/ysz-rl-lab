"""Claude07 safety regressions; all checks are local and never contact Slurm."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class SafetyTests(unittest.TestCase):
    def setUp(self):
        try:
            self.s = importlib.import_module('claude07_safety')
        except ImportError:
            self.fail('Claude07 safety implementation is missing')

    def test_budget_counts_failed_cost_and_remaining_without_double_counting(self):
        jobs = [dict(name='a', cap=3), dict(name='b', cap=5), dict(name='c', cap=1.5)]
        r = self.s.budget_projection(jobs, {'a': dict(state='RUNNING', actual=1),
                                           'b': dict(state='FAILED', actual=.1)}, 10)
        self.assertAlmostEqual(r['actual'], 11.1)
        self.assertAlmostEqual(r['remaining_caps'], 3.5)
        self.assertAlmostEqual(r['actual_plus_remaining_caps'], 14.6)

    def test_round_cap_and_existing_pool_cap_both_apply(self):
        jobs = [dict(name='a', cap=27.5)]
        self.assertTrue(self.s.budget_status(jobs, {}, 10.5380555556, 40, 27.5)['safe'])
        self.assertFalse(self.s.budget_status(jobs, {}, 13, 40, 27.5)['safe'])
        self.assertFalse(self.s.budget_status([dict(name='a', cap=27.5001)], {}, 0, 40, 27.5)['safe'])

    def test_unexpected_overrun_is_pool_risk_even_if_total_still_fits(self):
        r = self.s.budget_status([dict(name='a', cap=3)],
                                {'a': dict(state='FAILED', actual=3.01)}, 0, 40, 27.5)
        self.assertFalse(r['safe'])
        self.assertIn('job_cap_exceeded:a', r['reasons'])

    def test_unknown_or_negative_cost_is_not_silently_accepted(self):
        for observed in ({'extra': dict(actual=1)}, {'a': dict(actual=-1)}):
            with self.assertRaises(ValueError):
                self.s.budget_projection([dict(name='a', cap=1)], observed, 0)

    def test_failures_block_only_actual_dependencies(self):
        jobs = [dict(name='cpu', group='C1', depends_on=[]),
                dict(name='g1', group='G1', depends_on=[]),
                dict(name='g2a', group='G2', depends_on=[]),
                dict(name='g2b', group='G2', depends_on=[]),
                dict(name='dependent', group='G1', depends_on=['g1']),
                dict(name='transitive', group='G1', depends_on=['dependent'])]
        self.assertEqual(self.s.blocked_jobs(jobs, {'cpu'}), [])
        self.assertEqual(self.s.blocked_jobs(jobs, {'g2a'}), [])
        self.assertEqual(self.s.blocked_jobs(jobs, {'g1'}), ['dependent', 'transitive'])

    def test_resource_lane_uses_afterany_and_is_not_a_science_dependency(self):
        job = dict(name='g3', resource_afterany=['g1'], depends_on=[])
        self.assertEqual(self.s.dependency_flags(job, {'g1': '123'}), ['--dependency=afterany:123'])
        job['depends_on'] = ['relax']
        self.assertEqual(self.s.dependency_flags(job, {'g1': '123', 'relax': '456'}),
                         ['--dependency=afterany:123,afterok:456'])

    def test_last_closed_frame_ignores_incomplete_half_step(self):
        frame = ('<calculation><scstep/><structure><crystal><varray name="basis">'
                 '<v>2 0 0</v><v>0 2 0</v><v>0 0 2</v></varray></crystal>'
                 '<varray name="positions"><v>.1 .2 .3</v></varray></structure>'
                 '<varray name="forces"><v>.01 0 0</v></varray></calculation>')
        xml = '<modeling>' + frame + '<calculation><structure><varray name="positions"><v>.9 .9 .9</v>'
        r = self.s.last_complete_calculation(xml, expected_index=1)
        self.assertEqual(r['positions'], [[.1, .2, .3]])
        self.assertEqual(r['completed_calculations'], 1)
        self.assertTrue(r['trailing_xml_incomplete'])
        with self.assertRaises(ValueError):
            self.s.last_complete_calculation(xml, expected_index=108)

    def test_no_contcar_fallback_when_xml_has_no_complete_calculation(self):
        with self.assertRaises(ValueError):
            self.s.last_complete_calculation('<modeling><calculation><structure/>')

    def test_real_frame108_is_from_xml_with_verified_force(self):
        source = ROOT/'calibration_gpu/claude06_outputs/c06_L00_relax_fromideal'
        r = self.s.last_complete_calculation((source/'vasprun.xml').read_text(), 108)
        self.assertEqual(r['scf_steps'], 7)
        fmax = max(sum(x*x for x in v)**.5 for v in r['forces'])
        self.assertAlmostEqual(fmax, .053189480253179766)
        self.assertEqual(len(r['positions']), 316)

    def test_hash_verification_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d/'INCAR').write_text('NSW = 0\n')
            digest = hashlib.sha256((d/'INCAR').read_bytes()).hexdigest()
            (d/'REVIEWED_INPUTS.sha256').write_text(digest+'  INCAR\n')
            self.s.verify_hash_manifest(d)
            (d/'INCAR').write_text('NSW = 200\n')
            with self.assertRaises(ValueError):
                self.s.verify_hash_manifest(d)

    def test_hash_manifest_cannot_escape_task_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d/'REVIEWED_INPUTS.sha256').write_text('0'*64+'  ../INCAR\n')
            with self.assertRaises(ValueError):
                self.s.verify_hash_manifest(d)

    def test_task_fail_records_are_atomic_and_do_not_make_pool_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp)
            for name in ['a', 'b']:
                (pool/name).mkdir()
                self.s.record_task_status(pool/name, 'FAILED', 'independent failure', 124)
            for name in ['a', 'b']:
                r = json.loads((pool/name/'TASK_STATUS.json').read_text())
                self.assertEqual(r['state'], 'FAILED')
                self.assertEqual(r['exit_code'], 124)
            self.assertFalse((pool/'CLAUDE07_POOL_STOP.json').exists())
            self.assertEqual(list(pool.rglob('*.tmp')), [])

    def make_runtime_fixture(self, directory, depends_on=None):
        directory.mkdir()
        meta = dict(gpus=0, cores=32, mpi_ranks=32, depends_on=depends_on or [])
        (directory/'job.json').write_text(json.dumps(meta))
        (directory/'REVIEWED_INPUTS.sha256').write_text(
            hashlib.sha256((directory/'job.json').read_bytes()).hexdigest()+'  job.json\n')

    def test_old_stop_and_independent_failure_do_not_block_new_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp)
            (pool/'CLAUDE06_STOP.json').write_text('{broken old marker')
            self.make_runtime_fixture(pool/'failed')
            self.s.record_task_status(pool/'failed', 'FAILED', 'SCF', 93)
            self.make_runtime_fixture(pool/'independent')
            with patch.dict(os.environ, SLURM_NTASKS='32', SLURM_CPUS_PER_TASK='1'):
                self.assertEqual(self.s.claim_task(pool/'independent'), 0)
            self.assertEqual(json.loads((pool/'independent/TASK_STATUS.json').read_text())['state'], 'RUNNING')

    def test_true_dependency_failure_blocks_only_dependent(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp)
            self.make_runtime_fixture(pool/'failed')
            self.s.record_task_status(pool/'failed', 'FAILED', 'SCF', 93)
            self.make_runtime_fixture(pool/'dependent', ['failed'])
            self.assertEqual(self.s.claim_task(pool/'dependent'), 91)
            self.assertEqual(json.loads((pool/'dependent/TASK_STATUS.json').read_text())['state'], 'BLOCKED')
            self.assertFalse((pool/'CLAUDE07_POOL_STOP.json').exists())

    def test_existing_claim_and_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp)
            self.make_runtime_fixture(pool/'a')
            with patch.dict(os.environ, SLURM_NTASKS='32', SLURM_CPUS_PER_TASK='1'):
                self.assertEqual(self.s.claim_task(pool/'a'), 0)
                status = (pool/'a/TASK_STATUS.json').read_bytes()
                with self.assertRaises(FileExistsError):
                    self.s.claim_task(pool/'a')
                self.assertEqual((pool/'a/TASK_STATUS.json').read_bytes(), status)
                self.make_runtime_fixture(pool/'b')
                (pool/'b/OUTCAR').write_text('prior result must survive')
                with self.assertRaises(FileExistsError):
                    self.s.claim_task(pool/'b')
                self.assertEqual((pool/'b/OUTCAR').read_text(), 'prior result must survive')
                self.assertFalse((pool/'b/TASK_STATUS.json').exists())

    def test_pool_stop_is_budget_only_and_local_to_one_resource(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = Path(tmp)/'gpu'
            cpu = Path(tmp)/'cpu'
            gpu.mkdir(); cpu.mkdir()
            safe = self.s.budget_status([dict(name='a', cap=1)], {}, 0, 40, 27.5)
            with self.assertRaises(ValueError):
                self.s.write_budget_stop(gpu, safe)
            unsafe = self.s.budget_status([dict(name='a', cap=28)], {}, 0, 40, 27.5)
            self.s.write_budget_stop(gpu, unsafe)
            self.assertTrue(json.loads((gpu/'CLAUDE07_POOL_STOP.json').read_text())['reasons'])
            self.assertFalse((cpu/'CLAUDE07_POOL_STOP.json').exists())

    def test_resource_fingerprint_rejects_changed_mpi_allocation(self):
        meta = dict(gpus=1, cores=6, mpi_ranks=1)
        self.s.validate_allocation(meta, {'SLURM_NTASKS': '1', 'SLURM_CPUS_PER_TASK': '6'})
        with self.assertRaises(ValueError):
            self.s.validate_allocation(meta, {'SLURM_NTASKS': '2', 'SLURM_CPUS_PER_TASK': '6'})

    def test_incomplete_or_unconverged_output_is_failure(self):
        xml = '<modeling><parameters><i name="NELM">120</i></parameters><calculation><scstep/></calculation></modeling>'
        out = 'aborting loop because EDIFF is reached'
        self.assertTrue(self.s.validate_output(xml, out, False)['valid'])
        self.assertFalse(self.s.validate_output(xml, out, True)['valid'])
        self.assertFalse(self.s.validate_output(xml[:-10], out, False)['valid'])


class PreparedInputTests(unittest.TestCase):
    def test_missing_completed_xml_never_falls_back_to_contcar(self):
        prep = importlib.import_module('prepare_claude07')
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp)/'source'
            for relative in ['calibration_gpu/GPU_BUDGET.json', 'calibration/BUDGET.json',
                             'calibration_gpu/CLAUDE06_JOB_LIST.json', 'calibration/CLAUDE06_JOB_LIST.json',
                             'calibration_gpu/claude06_outputs/c06_L00_relax_fromideal/submitted_job_id.txt',
                             'calibration_gpu/claude06_outputs/c06_L00_relax_fromideal/CONTCAR']:
                (fake/relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/relative, fake/relative)
            xml = fake/'calibration_gpu/claude06_outputs/c06_L00_relax_fromideal/vasprun.xml'
            xml.write_text('<modeling><calculation><structure/>')
            with self.assertRaisesRegex(ValueError, 'complete calculation'):
                prep.prepare(fake, Path(tmp)/'output')
            self.assertFalse((Path(tmp)/'output').exists())

    def test_all_prepared_inputs_match_authorization_and_are_idempotent(self):
        try:
            prep = importlib.import_module('prepare_claude07')
        except ImportError:
            self.fail('Claude07 input preparation implementation is missing')
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            first = prep.prepare(ROOT, target)
            before = {str(f.relative_to(target)): hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in target.rglob('*') if f.is_file()}
            second = prep.prepare(ROOT, target)
            after = {str(f.relative_to(target)): hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in target.rglob('*') if f.is_file()}
            self.assertEqual(first, second)
            self.assertEqual(before, after)
            self.assertEqual(first['gpu_jobs'], 8)
            self.assertEqual(first['cpu_jobs'], 1)
            self.assertAlmostEqual(first['gpu_cap_card_hours'], 27.5)
            self.assertEqual(first['cpu_cap_core_hours'], 32)
            jobs = json.loads((target/'calibration_gpu/claude07/JOB_LIST.json').read_text())
            self.assertEqual([j['group'] for j in jobs].count('G1'), 1)
            self.assertEqual([j['group'] for j in jobs].count('G2'), 4)
            self.assertEqual([j['group'] for j in jobs].count('G3'), 3)
            self.assertTrue(all(j['depends_on'] == [] for j in jobs))
            for i, job in enumerate(jobs):
                self.assertEqual(job['lane'], i % 2)
                self.assertEqual(job['resource_afterany'], [jobs[i-2]['name']] if i >= 2 else [])
                d = target/'calibration_gpu/claude07/execution'/job['name']
                self.assertNotIn('CLAUDE06_STOP', (d/'sub.sh').read_text()+(d/'run.sh').read_text())
                self.assertEqual(job['gpus'], 1)
                self.assertEqual(job['cores'], 6)
                self.assertEqual(job['memory'], '90G')
                self.assertEqual(job['watchdog_seconds'], job['limit_seconds']-60)
                values = prep.incar_values((d/'INCAR').read_text())
                self.assertEqual(values['LREAL'], 'Auto')
                self.assertEqual(values['EDIFF'], '1E-5')
                self.assertEqual(values['ENCUT'], '520')
                self.assertEqual(values['NCORE'], '1')
                self.assertEqual(values['KPAR'], '1')
                if job['group'] == 'G3':
                    self.assertEqual(job['kind'], 'sp')
                    self.assertEqual(values['NSW'], '0')
                    self.assertEqual(job['binary'], 'vasp_std')
                    self.assertEqual(job['kmesh'], [2, 2, 2])
                    self.assertEqual((d/'POSCAR').read_bytes(),
                                     (ROOT/'calibration_gpu/execution_claude06'/('c06_P2b_'+job['config'])/'POSCAR').read_bytes())
                else:
                    self.assertEqual(values['NSW'], '200')
                    self.assertEqual(values['EDIFFG'], '-0.02')
                    self.assertEqual(job['binary'], 'vasp_gam')
                    self.assertEqual(job['kmesh'], [1, 1, 1])
            cpu = json.loads((target/'calibration/claude07/JOB_LIST.json').read_text())[0]
            self.assertEqual(cpu['limit_seconds'], 3600)
            self.assertEqual(cpu['watchdog_seconds'], 3540)
            self.assertEqual(cpu['mpi_ranks'], 32)
            for filename in ['POSCAR', 'KPOINTS']:
                self.assertEqual((target/cpu['local_directory']/filename).read_bytes(),
                                 (ROOT/'calibration/execution_claude06/c06_S08_k333'/filename).read_bytes())
            d = target/'calibration_gpu/claude07/execution'/jobs[0]['name']
            (d/'submitted_job_id.txt').write_text('999\n')
            with self.assertRaises(ValueError):
                prep.prepare(ROOT, target)
            self.assertEqual((d/'submitted_job_id.txt').read_text(), '999\n')


if __name__ == '__main__':
    unittest.main()
