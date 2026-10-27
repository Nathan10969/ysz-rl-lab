"""Exercise the generated submit transaction without SSH or real Slurm calls."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import submit_smoke_claude07 as submitter


class SmokeSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name) / 'project'
        self.root = self.project / 'training_smoke'
        self.dft_root = Path(self.temp.name) / 'dft'
        self.dft_root.mkdir(parents=True)
        self.jobs = [
            {'id': 'c07_smoke_sft', 'sub': 'jobs/sft/sub.sh',
             'gpus': 1, 'cpu_cores': 6, 'hard_gpu_hours': 0.5},
            {'id': 'c07_smoke_mace', 'sub': 'jobs/mace/sub.sh',
             'gpus': 1, 'cpu_cores': 6, 'hard_gpu_hours': 0.75},
        ]
        self.hashes = {}
        for job in self.jobs:
            path = self.root / job['sub']
            path.parent.mkdir(parents=True)
            path.write_text('#!/bin/sh\nexit 0\n')
            name = str(path.relative_to(self.project))
            self.hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest = {'remote_smoke_root': str(self.root), 'jobs': self.jobs}
        self.budget = {'authorized_card_hours': 10, 'actual_card_hours': 0,
                       'actual_plus_remaining_caps': 1.25}
        self.dft = {'jobs': [
            {'slurm_job_id': str(sid)} for sid in (617, 619, 621, 623, 625, 627, 629, 631)
        ]}
        (self.dft_root / 'SUBMISSIONS.json').write_text(json.dumps(self.dft))
        self.calls = []
        self.call_lock = threading.Lock()

    def code(self, nonce='test-claim'):
        with patch.object(submitter, 'DFT_REMOTE', str(self.dft_root)):
            return submitter.remote_code(self.manifest, self.budget, self.hashes,
                                         self.dft, ['629', '631'], nonce)

    def ledger(self):
        return json.loads((self.root / 'SUBMISSIONS.json').read_text())

    def slurm(self, args, **kwargs):
        if '--test-only' in args:
            return types.SimpleNamespace(returncode=0, stdout=b'', stderr=b'')
        self.assertEqual(args[:2], ['sbatch', '--parsable'])
        # The intent must already be durable before the external side effect.
        ledger = self.ledger()
        self.assertIn('pending_submission', ledger)
        with self.call_lock:
            self.calls.append(args)
            sid = 700 + len(self.calls)
        return types.SimpleNamespace(returncode=0, stdout=str(sid).encode(), stderr=b'')

    def execute(self, code=None, run=None, queue=None):
        with patch.object(subprocess, 'check_output', queue or (lambda *a, **k: b'')), \
                patch.object(subprocess, 'run', run or self.slurm):
            exec(compile(code or self.code(), 'remote_smoke_submission', 'exec'), {})

    def test_success_records_two_jobs_behind_both_dft_tails(self):
        self.execute()
        ledger = self.ledger()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual([j['slurm_job_id'] for j in ledger['jobs']], ['701', '702'])
        self.assertEqual(ledger['DFT_lane_tails'], ['629', '631'])
        self.assertEqual(ledger['pool'], 'TRAINING_SMOKE')
        self.assertNotIn('pending_submission', ledger)
        for args in self.calls:
            self.assertIn('--dependency=afterany:629:631', args)
            self.assertNotIn('--dependency=afterok:629:631', args)
        with self.assertRaises(RuntimeError):
            self.execute(self.code('another-runner'))
        self.assertEqual(len(self.calls), 2)

    def test_ambiguous_sbatch_retains_intent_and_prevents_retry(self):
        def accepted_but_lost(args, **kwargs):
            response = self.slurm(args, **kwargs)
            if '--parsable' in args:
                raise subprocess.TimeoutExpired(args, 1)
            return response
        with self.assertRaises(subprocess.TimeoutExpired):
            self.execute(run=accepted_but_lost)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.ledger()['pending_submission']['id'], 'c07_smoke_sft')
        self.assertEqual(self.ledger()['jobs'], [])
        self.assertFalse((self.root / 'jobs/sft/submitted_job_id.txt').exists())
        with self.assertRaises(RuntimeError):
            self.execute(self.code('new-nonce'))
        self.assertEqual(len(self.calls), 1)

    def test_duplicate_slurm_id_retains_second_intent(self):
        def duplicate(args, **kwargs):
            response = self.slurm(args, **kwargs)
            if '--parsable' in args:
                response.stdout = b'701'
            return response
        with self.assertRaisesRegex(RuntimeError, 'Duplicate Slurm ID'):
            self.execute(run=duplicate)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.ledger()['jobs']), 1)
        self.assertEqual(self.ledger()['pending_submission']['id'], 'c07_smoke_mace')

    def test_changed_remote_dft_ledger_prevents_submission(self):
        (self.dft_root / 'SUBMISSIONS.json').write_text(json.dumps({'jobs': []}))
        with self.assertRaisesRegex(RuntimeError, 'DFT ledger changed'):
            self.execute()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / 'SUBMISSION_CLAIM.json').exists())

    def test_changed_remote_input_prevents_submission(self):
        (self.root / 'jobs/sft/sub.sh').write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'Remote input hash changed'):
            self.execute()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / 'SUBMISSION_CLAIM.json').exists())

    def test_relevant_pool_stop_prevents_submission(self):
        (self.root / 'STOP_SMOKE.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'Smoke pool stop'):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_two_runners_have_only_one_successful_claim(self):
        barrier = threading.Barrier(2, timeout=5)
        failures = []
        def queue(*args, **kwargs):
            barrier.wait()
            return b''
        def runner(code):
            try:
                exec(compile(code, 'remote_smoke_submission', 'exec'), {})
            except BaseException as error:
                failures.append(error)
        with patch.object(subprocess, 'check_output', queue), \
                patch.object(subprocess, 'run', self.slurm):
            threads = [threading.Thread(target=runner, args=(self.code(nonce),))
                       for nonce in ('runner-A', 'runner-B')]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
            self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], FileExistsError)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.ledger()['jobs']), 2)

    def test_safety_checks_survive_optimization_and_cli_requires_submit(self):
        source = Path(submitter.__file__).read_text()
        self.assertFalse(any(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(source))))
        self.assertFalse(any(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(self.code()))))
        module = ast.parse(source)
        main = module.body[-1]
        module.body = [ast.Import(names=[ast.alias(name='argparse')]), main]
        for node in ast.walk(module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == 'parse_args':
                node.args = [ast.List(elts=[], ctx=ast.Load())]
        ast.fix_missing_locations(module)
        for optimize in (0, 2):
            calls = []
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                exec(compile(module, 'cli_guard', 'exec', optimize=optimize),
                     {'__name__': '__main__', 'submit': lambda: calls.append(True)})
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
