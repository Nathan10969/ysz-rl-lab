"""Execute the frozen remote wrapper with fake scheduler calls, never a real submit."""
import json
import hashlib
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]


class SubmissionTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory).resolve() / 'pool/claude10/teacher'
        root.mkdir(parents=True)
        for name in ('submit_once.py', 'JOB_MANIFEST.json', 'SUBMISSION_INTENT_ID.json',
                     'AUTHORIZATION.json', 'POOL_PRE_SUBMISSION.json'):
            shutil.copy2(PROJECT / 'training_smoke/claude10/teacher' / name, root / name)
        manifest = json.loads((root/'JOB_MANIFEST.json').read_text())
        manifest['remote_smoke_root'] = str(root)
        (root/'JOB_MANIFEST.json').write_text(json.dumps(manifest))
        sources = {}
        for name in ('claude10_physics.py','claude08_env.py','smoke_mace.py','smoke_grpo08.py'):
            path = root/'project/scripts'/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('# mock-only source, never executed\n')
            sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        evidence = root/'cpu_preflight/result.json'
        evidence.parent.mkdir()
        evidence.write_text('{"scope":"mock-only, not real CPU physics"}')
        (root/'CPU_PREFLIGHT_VERIFIED.json').write_text(json.dumps(dict(independently_verified=True,
            CPU_only=True, same_production_code_path=True, batch_sizes=[1,4], raw_ef=22,
            source_sha256=sources,
            checkpoint_sha256='d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9',
            evidence_sha256={'cpu_preflight/result.json':hashlib.sha256(evidence.read_bytes()).hexdigest()})))
        return root

    def run_mocked(self, root, *, response='701\n', check_code=0):
        calls = []
        def fake_run(args, **kwargs):
            calls.append(args)
            if args[0] == 'squeue':
                return subprocess.CompletedProcess(args, 0, '699|ysz_c10_old|RUNNING\n800|student_other|RUNNING\n', '')
            if '--test-only' in args:
                return subprocess.CompletedProcess(args, check_code, '', 'rejected' if check_code else '')
            return subprocess.CompletedProcess(args, 0, response if '--parsable' in args else '', '')
        with patch('subprocess.run', side_effect=fake_run):
            runpy.run_path(str(root/'submit_once.py'))
        return calls

    def test_success_uses_one_actual_submit_and_project_serial_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            calls = self.run_mocked(root)
            actual = [a for a in calls if '--parsable' in a]
            self.assertEqual(len(actual), 1)
            self.assertIn('--dependency=afterany:699', actual[0])
            self.assertNotIn('800', ' '.join(actual[0]))
            ledger = json.loads((root/'SUBMISSIONS.json').read_text())
            self.assertEqual(ledger['jobs'][0]['slurm_job_id'], '701')
            self.assertNotIn('pending_submission', ledger)
            with patch('subprocess.run') as scheduler:
                with self.assertRaises(RuntimeError): runpy.run_path(str(root/'submit_once.py'))
                scheduler.assert_not_called()

    def test_ambiguous_response_keeps_claim_and_intent_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            with self.assertRaises(RuntimeError): self.run_mocked(root, response='UNKNOWN')
            self.assertTrue((root/'SUBMISSION_CLAIM.json').exists())
            ledger = json.loads((root/'SUBMISSIONS.json').read_text())
            self.assertEqual(ledger['pending_submission'], 'c10_teacher')
            self.assertEqual(ledger['jobs'], [])
            with patch('subprocess.run') as scheduler:
                with self.assertRaises(RuntimeError): runpy.run_path(str(root/'submit_once.py'))
                scheduler.assert_not_called()

    def test_failed_scheduler_validation_never_consumes_gpu_submit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            with self.assertRaises(RuntimeError): self.run_mocked(root, check_code=1)
            self.assertFalse((root/'SUBMISSION_CLAIM.json').exists())
            self.assertFalse((root/'SUBMISSIONS.json').exists())

    def test_missing_cpu_execution_gate_prevents_scheduler_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            (root/'CPU_PREFLIGHT_VERIFIED.json').write_text('{}')
            with patch('subprocess.run') as scheduler:
                with self.assertRaises(ValueError): runpy.run_path(str(root/'submit_once.py'))
                self.assertEqual(scheduler.call_count, 1)  # only frozen-file SHA verification
                self.assertEqual(scheduler.call_args.args[0][0], 'sha256sum')


if __name__ == '__main__': unittest.main()
