"""Offline tests: no SSH, Slurm submission, model load, or GPU invocation."""
import importlib
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def prepared_budget():
    return dict(authorized_card_hours=10, actual_card_hours=0, actual_gpu_hours=0,
                actual_plus_remaining_caps=1.25, per_job_gpu_hour_cap=2,
                jobs=[dict(id='c07_smoke_sft', gpus=1, hard_gpu_hours=.5),
                      dict(id='c07_smoke_mace', gpus=1, hard_gpu_hours=.75)])


def manifest():
    return dict(schema='ysz.training_smoke.v1', remote_smoke_root='/remote/training_smoke',
                jobs=[dict(id='c07_smoke_sft', gpus=1, cpu_cores=6, hard_gpu_hours=.5,
                           seen_token_cap=32768, optimization_steps=32),
                      dict(id='c07_smoke_mace', gpus=1, cpu_cores=6, hard_gpu_hours=.75,
                           raw_ef_call_cap=1024, relax_steps_per_cell_cap=40)])


def sft_result():
    return dict(status='completed', seen_tokens=100, action_tokens=20, optimizer_steps=32,
                microbatches=64, validation_n=4, generation_tokens=256,
                adapter_reload=dict(unloaded_without_merge=True, reloaded_from_disk=True, state_exact_match=True),
                wall_seconds=10, runtime={'job_id': '1', 'visible_gpus': 1})


def mace_result():
    return dict(status='completed', raw_ef_calls=20, energy_calls=20, force_calls=20,
                model_forward_calls=10, wall_seconds=4, runtime={'job_id': '2', 'visible_gpus': 1},
                single_batch_SP={str(n): {'cells_per_second': 1} for n in [1, 2, 4]},
                batch_equivalence={'batch_sizes_checked': [2, 4], 'per_batch_size': {str(n):
                    {'max_energy_difference_eV': 1e-8, 'max_force_difference_eV_A': 1e-8, 'passed': True} for n in [2, 4]}},
                relaxations=[dict(steps=40) for _ in range(6)])


class SmokeCollectionTests(unittest.TestCase):
    def cached_635_observations(self):
        text, proof = self.real_635_allocation()
        ledger = {'jobs': [dict(id='c07_smoke_mace', slurm_job_id='635')]}
        old = dict(checked_at_utc='2026-10-24T04:56:56Z', remote_ledger=ledger,
                   account={'code': 0, 'stdout': text+'\n'}, allocation_proofs={'635': proof})
        fresh = dict(checked_at_utc='2026-10-24T05:10:00Z', remote_ledger=ledger,
                     account={'code': 0, 'stdout': text+'\n'}, allocation_proofs={'635': dict(
                         code=1, stdout='', stderr='slurm_load_jobs error: Invalid job id specified', source='new_snapshot#635')})
        return old, fresh, ledger

    def test_purged_scontrol_reuses_bound_terminal_proof_and_keeps_original_provenance(self):
        old, fresh, ledger = self.cached_635_observations()
        result = self.c.preserve_terminal_allocation_proofs(fresh, old, manifest())
        proof = result['allocation_proofs']['635']
        self.assertEqual(proof['code'], 0)
        self.assertEqual(proof['stdout'], old['allocation_proofs']['635']['stdout'])
        self.assertEqual(proof['source'], old['allocation_proofs']['635']['source'])
        self.assertEqual(proof['captured_at_utc'], old['checked_at_utc'])
        self.assertEqual(proof['stdout_sha256'], hashlib.sha256(proof['stdout'].encode()).hexdigest())
        self.assertEqual(result['fresh_allocation_proofs']['635']['code'], 1)
        self.assertTrue(proof['cache_reuse']['same_terminal_parent_binding'])
        rows, risks, unknown = self.c.parse_accounting(result['account']['stdout'], manifest(), ledger,
            prepared_budget(), allocation_proofs=result['allocation_proofs'])
        self.assertTrue(rows['635']['allocation_verified'])
        self.assertAlmostEqual(rows['635']['actual_card_hours'], .0025)
        self.assertEqual(rows['635']['NTasks_fallback_evidence']['captured_at_utc'], old['checked_at_utc'])
        self.assertTrue(rows['635']['NTasks_fallback_evidence']['cache_reuse']['same_terminal_parent_binding'])
        self.assertFalse(risks)
        self.assertFalse(unknown)

    def test_cached_proof_never_repairs_changed_parent_identity_time_state_or_tres(self):
        import copy
        old, current, ledger = self.cached_635_observations()
        changes = [('635|', '636|'), ('|9|', '|10|'), ('|FAILED|', '|COMPLETED|'),
                   ('|FAILED|', '|RUNNING|'), ('cpu=6,', 'cpu=12,'), ('mem=90G', 'mem=91G'),
                   ('gres/gpu:a100=1,gres/gpu=1', 'gres/gpu:a100=2,gres/gpu=2')]
        for before, after in changes:
            with self.subTest(change=after):
                fresh = copy.deepcopy(current)
                fresh['account']['stdout'] = fresh['account']['stdout'].replace(before, after)
                result = self.c.preserve_terminal_allocation_proofs(fresh, old, manifest())
                self.assertEqual(result['allocation_proofs']['635']['code'], 1)
                rows, risks, unknown = self.c.parse_accounting(result['account']['stdout'], manifest(), ledger,
                    prepared_budget(), allocation_proofs=result['allocation_proofs'])
                self.assertTrue(unknown or risks)
                self.assertFalse(rows.get('635', {}).get('allocation_verified', False))

    def test_fresh_conflicting_successful_proof_is_never_hidden_by_cache(self):
        old, fresh, ledger = self.cached_635_observations()
        fresh['allocation_proofs']['635'] = dict(old['allocation_proofs']['635'],
            stdout=old['allocation_proofs']['635']['stdout'].replace('NumTasks=1', 'NumTasks=2'))
        result = self.c.preserve_terminal_allocation_proofs(fresh, old, manifest())
        self.assertIn('NumTasks=2', result['allocation_proofs']['635']['stdout'])
        self.assertNotIn('635', result['allocation_proof_cache'])
        rows, risks, unknown = self.c.parse_accounting(result['account']['stdout'], manifest(), ledger,
            prepared_budget(), allocation_proofs=result['allocation_proofs'])
        self.assertIn('635', unknown)
        self.assertTrue(risks)
        # A failed command containing resource fields is also ambiguous; it
        # neither proves a new allocation nor permits hiding it behind cache.
        fresh['allocation_proofs']['635']['code'] = 1
        result = self.c.preserve_terminal_allocation_proofs(fresh, old, manifest())
        self.assertEqual(result['allocation_proofs']['635']['code'], 1)
        self.assertNotIn('635', result['allocation_proof_cache'])
        rows, risks, unknown = self.c.parse_accounting(result['account']['stdout'], manifest(), ledger,
            prepared_budget(), allocation_proofs=result['allocation_proofs'])
        self.assertIn('635', unknown)
        self.assertFalse(risks)

    def test_transient_account_failure_keeps_cache_but_cannot_use_it_as_current_account(self):
        import copy
        old, fresh, ledger = self.cached_635_observations()
        failed_query = copy.deepcopy(fresh)
        failed_query['account'] = dict(code=1, stdout='', stderr='temporary accounting service failure')
        failed_query['allocation_proofs'] = {}
        deferred = self.c.preserve_terminal_allocation_proofs(failed_query, old, manifest())
        self.assertEqual(deferred['allocation_proofs'], {})
        self.assertIn('635', deferred['allocation_proof_cache'])
        recovered = self.c.preserve_terminal_allocation_proofs(fresh, deferred, manifest())
        self.assertEqual(recovered['allocation_proofs']['635']['code'], 0)

    def test_stderr_resource_conflict_cannot_be_hidden_by_a_purge_diagnostic(self):
        old, fresh, ledger = self.cached_635_observations()
        fresh['allocation_proofs']['635']['stderr'] = ('Invalid job id specified\n'
            'JobId=635 NumCPUs=12 NumTasks=2 CPUs/Task=6 TRES=cpu=12,gres/gpu=2')
        result = self.c.preserve_terminal_allocation_proofs(fresh, old, manifest())
        self.assertEqual(result['allocation_proofs']['635']['code'], 1)
        self.assertEqual(result['allocation_proofs']['635']['stderr'], fresh['allocation_proofs']['635']['stderr'])
        self.assertNotIn('635', result['allocation_proof_cache'])
        rows, risks, unknown = self.c.parse_accounting(result['account']['stdout'], manifest(), ledger,
            prepared_budget(), allocation_proofs=result['allocation_proofs'])
        self.assertIn('635', unknown)
        self.assertFalse(rows['635']['verified'])
        self.assertFalse(risks)

    def test_missing_command_auth_network_and_unrecognized_exit_do_not_reuse_cache(self):
        old, purged, ledger = self.cached_635_observations()
        for code, error in [(127, 'scontrol: command not found'), (255, 'Permission denied'),
                            (1, 'Authentication failed'), (1, 'Connection timed out'),
                            (1, 'Permission denied\nInvalid job id specified'),
                            (2, 'Invalid job id specified')]:
            with self.subTest(code=code, error=error):
                import copy
                fresh = copy.deepcopy(purged)
                fresh['allocation_proofs']['635'].update(code=code, stderr=error)
                result = self.c.preserve_terminal_allocation_proofs(fresh, old, manifest())
                self.assertEqual(result['allocation_proofs']['635']['code'], code)
                self.assertIn('635', result['allocation_proof_cache'])
                rows, risks, unknown = self.c.parse_accounting(result['account']['stdout'], manifest(), ledger,
                    prepared_budget(), allocation_proofs=result['allocation_proofs'])
                self.assertIn('635', unknown)
                self.assertFalse(rows['635']['verified'])
                self.assertFalse(risks)
                recovered = self.c.preserve_terminal_allocation_proofs(purged, result, manifest())
                self.assertEqual(recovered['allocation_proofs']['635']['code'], 0)

    def real_635_allocation(self):
        path = Path(__file__).resolve().parents[1]/'training_smoke/evidence/635_start_failure_readonly_recheck.json'
        saved = json.loads(path.read_text())
        row = next(line for line in saved['account']['stdout'].splitlines() if line.startswith('635|'))
        proof = dict(saved['allocation635'], source=str(path)+'#allocation635')
        return row, proof

    def test_real_blank_ntasks_uses_matching_same_parent_proof_without_editing_raw(self):
        text, proof = self.real_635_allocation()
        ledger = {'jobs': [dict(id='c07_smoke_mace', slurm_job_id='635')]}
        rows, risks, unknown = self.c.parse_accounting(text, manifest(), ledger, prepared_budget(),
                                                      allocation_proofs={'635': proof})
        row = rows['635']
        self.assertTrue(row['verified'])
        self.assertTrue(row['allocation_verified'])
        self.assertAlmostEqual(row['actual_card_hours'], .0025)
        self.assertEqual(row['raw_fields'], text.split('|'))
        self.assertEqual(row['raw_fields'][6], '')
        evidence = row['NTasks_fallback_evidence']
        self.assertEqual(evidence['source'], proof['source'])
        self.assertEqual(evidence['stdout_sha256'], hashlib.sha256(proof['stdout'].encode()).hexdigest())
        self.assertEqual(evidence['parsed_allocation']['NumTasks'], '1')
        self.assertEqual(evidence['raw_stdout'], proof['stdout'])
        self.assertFalse(risks)
        self.assertFalse(unknown)

    def test_blank_ntasks_without_proof_remains_unknown(self):
        text, _ = self.real_635_allocation()
        rows, risks, unknown = self.c.parse_accounting(text, manifest(),
            {'jobs': [dict(id='c07_smoke_mace', slurm_job_id='635')]}, prepared_budget())
        self.assertFalse(rows['635']['verified'])
        self.assertIn('635', unknown)
        self.assertFalse(risks)
        self.assertNotIn('NTasks_fallback_evidence', rows['635'])

    def test_query_captures_one_same_observation_proof_only_for_charged_blank_ntasks(self):
        import os
        import shlex
        import subprocess
        text, proof = self.real_635_allocation()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); home = root/'local'; home.mkdir(); remote = root/'remote'; remote.mkdir()
            ledger = {'jobs': [dict(id='c07_smoke_sft', slurm_job_id='633'), dict(id='c07_smoke_mace', slurm_job_id='635')]}
            (home/'SUBMISSIONS.json').write_text(json.dumps(ledger))
            (remote/'SUBMISSIONS.json').write_text(json.dumps(ledger))
            account = root/'account.txt'; account.write_text('633|PENDING|0||0:0|6|\n'+text+'\n')
            raw = root/'allocation.txt'; raw.write_text(proof['stdout'])
            commands = root/'commands.txt'; binary = root/'bin'; binary.mkdir()
            scripts = {
                'sacct': 'printf "sacct\\n" >> '+shlex.quote(str(commands))+'\ncat '+shlex.quote(str(account)),
                'squeue': 'printf "squeue\\n" >> '+shlex.quote(str(commands))+'\nprintf "633|PENDING\\n"',
                'scontrol': 'printf "scontrol %s\\n" "$*" >> '+shlex.quote(str(commands))+'\ncat '+shlex.quote(str(raw))}
            for name, body in scripts.items():
                path = binary/name; path.write_text('#!/bin/sh\n'+body+'\n'); path.chmod(0o755)
            spec = manifest(); spec['remote_smoke_root'] = str(remote)
            def fake_ssh(host, command, payload, timeout):
                return subprocess.run(['python3', '-'], input=payload, capture_output=True, check=True,
                    env=dict(os.environ, PATH=str(binary)+':'+os.environ['PATH']))
            with patch.object(self.c, 'ssh', fake_ssh):
                snapshot = self.c.query(home, spec, 'fake')
            self.assertEqual(set(snapshot['allocation_proofs']), {'635'})
            captured = snapshot['allocation_proofs']['635']
            self.assertEqual(captured['stdout'], proof['stdout'])
            self.assertIn('/evidence/snapshot_', captured['source'])
            self.assertEqual(commands.read_text().splitlines(), ['sacct', 'squeue', 'scontrol show job 635 -o'])
            # A later regular observation must survive Slurm's purge of the
            # terminal job while preserving the original source and timestamp.
            original_source = captured['source']
            (binary/'scontrol').write_text('#!/bin/sh\nprintf "Invalid job id specified\\n" >&2\nexit 1\n')
            with patch.object(self.c, 'ssh', fake_ssh):
                after_purge = self.c.query(home, spec, 'fake')
            reused = after_purge['allocation_proofs']['635']
            self.assertEqual(reused['code'], 0)
            self.assertEqual(reused['source'], original_source)
            self.assertEqual(reused['captured_at_utc'], captured['captured_at_utc'])
            self.assertEqual(after_purge['fresh_allocation_proofs']['635']['code'], 1)

    def test_allocation_proof_wrong_id_or_resources_never_fill_sacct_fields(self):
        text, proof = self.real_635_allocation()
        changes = [('JobId=635', 'JobId=634', False), ('NumCPUs=6', 'NumCPUs=12', True),
                   ('NumTasks=1', 'NumTasks=2', True), ('CPUs/Task=6', 'CPUs/Task=3', True),
                   ('TRES=cpu=6,', 'TRES=cpu=12,', True), ('gres/gpu=1,gres/gpu:a100=1', 'gres/gpu=2,gres/gpu:a100=2', True)]
        for before, after, resource_risk in changes:
            with self.subTest(change=after):
                altered = dict(proof, stdout=proof['stdout'].replace(before, after))
                rows, risks, unknown = self.c.parse_accounting(text, manifest(),
                    {'jobs': [dict(id='c07_smoke_mace', slurm_job_id='635')]}, prepared_budget(),
                    allocation_proofs={'635': altered})
                self.assertFalse(rows['635']['verified'])
                self.assertIn('635', unknown)
                self.assertEqual(rows['635']['raw_fields'], text.split('|'))
                self.assertNotIn('NTasks_fallback_evidence', rows['635'])
                self.assertEqual(bool(risks), resource_risk)

    def test_ntasks_proof_never_repairs_missing_cpu_gpu_or_conflicting_tasks(self):
        text, proof = self.real_635_allocation()
        changed_rows = [text.replace('|6|', '||'), text.replace('gres/gpu:a100=1,gres/gpu=1,', ''),
                        text+'2', text.replace('gres/gpu:a100=1,gres/gpu=1,', 'gres/gpu:a100=2,gres/gpu=1,')]
        for changed in changed_rows:
            with self.subTest(row=changed):
                rows, risks, unknown = self.c.parse_accounting(changed, manifest(),
                    {'jobs': [dict(id='c07_smoke_mace', slurm_job_id='635')]}, prepared_budget(),
                    allocation_proofs={'635': proof})
                self.assertFalse(rows['635']['allocation_verified'])
                self.assertNotIn('NTasks_fallback_evidence', rows['635'])
                self.assertEqual(rows['635']['raw_fields'], changed.split('|'))

    def test_offline_failed_635_matching_proof_bills_parent_only_without_guarding_633(self):
        text, proof = self.real_635_allocation()
        ledger = {'jobs': [dict(id='c07_smoke_sft', slurm_job_id='633'), dict(id='c07_smoke_mace', slurm_job_id='635')]}
        for include_proof in [False, True]:
            with self.subTest(proof=include_proof), tempfile.TemporaryDirectory() as tmp:
                home = Path(tmp)
                snapshot = dict(checked_at_utc='saved fixture', remote_ledger=ledger, stop=None,
                    account=dict(code=0, stdout='633|PENDING|0||0:0|6|\n'+text+'\n'),
                    queue=dict(code=0, stdout='633|PENDING\n'),
                    allocation_proofs={'635': proof} if include_proof else {})
                for name, value in [('BUDGET.json', prepared_budget()), ('JOB_MANIFEST.json', manifest()),
                                    ('SUBMISSIONS.json', ledger), ('LATEST.json', snapshot)]:
                    (home/name).write_text(json.dumps(value))
                before = (home/'BUDGET.json').read_bytes()
                with patch.object(self.c, 'ssh', side_effect=AssertionError('Offline reconciliation must not query')):
                    analysis = self.c.collect(home, 'fake', offline=True)
                self.assertFalse(analysis['guard_plan']['pool_stop_required'])
                self.assertEqual(analysis['guard_plan']['cancel_ids'], [])
                self.assertFalse((home/'STOP_SMOKE.json').exists())
                if include_proof:
                    self.assertEqual(analysis['collection_status'], 'COLLECTED')
                    self.assertAlmostEqual(json.loads((home/'BUDGET.json').read_text())['actual_card_hours'], .0025)
                    self.assertEqual(next(r for r in analysis['jobs'] if r['slurm_job_id']=='635')['state'], 'FAILED')
                else:
                    self.assertEqual(analysis['collection_status'], 'DEFERRED')
                    self.assertEqual((home/'BUDGET.json').read_bytes(), before)

    def test_authorized_batch16_revision_requires_batch16_parity_and_all_relaxations(self):
        job = dict(manifest()['jobs'][1], batch_sizes=[1, 4, 16], relaxation_cells=21)
        result = mace_result()
        self.assertFalse(self.c.validate_result(job, result, '2')['valid'])
        result['single_batch_SP'] = {str(n): {'cells_per_second': 1} for n in [1, 4, 16]}
        result['batch_equivalence'] = {'batch_sizes_checked': [4, 16], 'per_batch_size': {
            str(n): {'max_energy_difference_eV': 1e-8, 'max_force_difference_eV_A': 1e-8,
                     'passed': True} for n in [4, 16]}}
        result['relaxations'] = [dict(steps=40) for _ in range(21)]
        self.assertTrue(self.c.validate_result(job, result, '2')['valid'])
        del result['batch_equivalence']['per_batch_size']['16']
        self.assertFalse(self.c.validate_result(job, result, '2')['valid'])

    def setUp(self):
        try:
            self.c = importlib.import_module('collect_claude07_smoke')
        except ImportError:
            self.fail('Smoke read-only collection orchestrator is missing')

    def test_ledgers_recover_remote_only_ids_but_reject_conflicts(self):
        local = {'jobs': [dict(id='c07_smoke_sft', slurm_job_id='1')]}
        remote = {'jobs': local['jobs']+[dict(id='c07_smoke_mace', slurm_job_id='2')]}
        self.assertEqual(len(self.c.reconcile_ledgers(manifest(), local, remote)['jobs']), 2)
        with self.assertRaises(ValueError):
            self.c.reconcile_ledgers(manifest(), local, {'jobs': [dict(id='c07_smoke_sft', slurm_job_id='9')]})
        with self.assertRaises(ValueError):
            self.c.reconcile_ledgers(manifest(), {'jobs': []}, {'jobs': [dict(id='unknown', slurm_job_id='8')]})

    def test_two_alias_tres_fields_are_one_gpu_not_two(self):
        ledger = {'jobs': [dict(id='c07_smoke_sft', slurm_job_id='1')]}
        records, risks, unknown = self.c.parse_accounting(
            '1|COMPLETED|360|cpu=6,gres/gpu=1,gres/gpu:a100=1|0:0|6|1\n', manifest(), ledger, prepared_budget())
        self.assertFalse(risks)
        self.assertFalse(unknown)
        self.assertAlmostEqual(records['1']['actual_card_hours'], .1)

    def test_allocated_two_gpus_is_billed_but_cannot_be_used_as_single_gpu_timing(self):
        ledger = {'jobs': [dict(id='c07_smoke_sft', slurm_job_id='1')]}
        records, risks, unknown = self.c.parse_accounting(
            '1|COMPLETED|360|cpu=6,gres/gpu=2|0:0|6|1\n', manifest(), ledger, prepared_budget())
        self.assertAlmostEqual(records['1']['actual_card_hours'], .2)
        self.assertFalse(records['1']['allocation_verified'])
        self.assertTrue(any('unexpected_GPU_allocation' in r for r in risks))

    def test_missing_allocation_after_elapsed_is_unknown_and_not_free(self):
        ledger = {'jobs': [dict(id='c07_smoke_sft', slurm_job_id='1')]}
        records, risks, unknown = self.c.parse_accounting(
            '1|FAILED|360|cpu=6|124:0|6|1\n', manifest(), ledger, prepared_budget())
        self.assertFalse(risks)
        self.assertIn('missing_GPU_allocation', records['1']['accounting_issues'])
        self.assertIsNone(records['1']['actual_card_hours'])
        self.assertIn('1', unknown)

    def test_helper_keeps_failed_cost_and_missing_job_known_cost_and_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            budget = prepared_budget()
            budget['jobs'][1].update(slurm_job_id='2', actual_card_hours=.2, state='RUNNING')
            (home/'BUDGET.json').write_text(json.dumps(budget))
            ledger = {'jobs': [dict(id='c07_smoke_sft', slurm_job_id='1'), dict(id='c07_smoke_mace', slurm_job_id='2')]}
            records, risks, unknown = self.c.parse_accounting(
                '1|TIMEOUT|900|cpu=6,gres/gpu=1|124:0|6|1\n', manifest(), ledger, budget)
            result = self.c.reconcile_budget(home, ledger, records, unknown, 'fixture')
            self.assertAlmostEqual(result['actual_card_hours'], .45)
            self.assertAlmostEqual(result['actual_plus_remaining_caps'], 1.0)
            self.assertEqual(result['accounting_unknown_job_ids'], ['2'])

    def test_accounting_regression_does_not_reduce_known_billed_cost(self):
        budget = prepared_budget()
        budget['jobs'][0].update(slurm_job_id='1', actual_card_hours=.2)
        records, risks, unknown = self.c.parse_accounting(
            '1|RUNNING|100|cpu=6,gres/gpu=1|0:0|6|1\n', manifest(),
            {'jobs': [dict(id='c07_smoke_sft', slurm_job_id='1')]}, budget)
        self.assertFalse(risks)
        self.assertIn('accounting_regression', records['1']['accounting_issues'])
        self.assertIn('1', unknown)

    def test_ordinary_failure_has_no_dependents_or_pending_cancellations(self):
        rows = [dict(id='c07_smoke_sft', slurm_job_id='1', state='FAILED'),
                dict(id='c07_smoke_mace', slurm_job_id='2', state='PENDING')]
        plan = self.c.guard_plan(rows, prepared_budget(), [], None)
        self.assertFalse(plan['pool_stop_required'])
        self.assertEqual(plan['cancel_ids'], [])

    def test_pool_risk_targets_only_these_pending_ids_never_running(self):
        rows = [dict(id='c07_smoke_sft', slurm_job_id='1', state='RUNNING'),
                dict(id='c07_smoke_mace', slurm_job_id='2', state='PENDING')]
        plan = self.c.guard_plan(rows, prepared_budget(), ['unexpected_GPU_allocation:1'], None)
        self.assertEqual(plan['cancel_ids'], ['2'])
        self.assertTrue(plan['pool_stop_required'])

    def test_corrupt_stop_is_present_and_raw_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'STOP_SMOKE.json'
            path.write_text('{bad\n')
            marker = self.c.read_marker(path)
            plan = self.c.guard_plan([], prepared_budget(), [], marker)
            self.assertTrue(plan['pool_stop_required'])
            self.assertEqual(marker['raw_text'], '{bad\n')

    def test_sft_completed_metrics_must_be_finite_and_bounded(self):
        job = manifest()['jobs'][0]
        result = sft_result()
        self.assertTrue(self.c.validate_result(job, result, '1')['valid'])
        result['wall_seconds'] = float('nan')
        self.assertFalse(self.c.validate_result(job, result, '1')['valid'])
        result['wall_seconds'] = 10; result['seen_tokens'] = 32769
        self.assertTrue(self.c.validate_result(job, result, '1')['consumption_risks'])

    def test_mace_missing_status_or_metrics_cannot_be_used(self):
        job = manifest()['jobs'][1]
        self.assertFalse(self.c.validate_result(job, {}, '2')['valid'])
        result = mace_result()
        self.assertTrue(self.c.validate_result(job, result, '2')['valid'])
        result['force_calls'] = 19
        self.assertFalse(self.c.validate_result(job, result, '2')['valid'])

    def test_missing_reload_or_parity_independent_checks_are_unusable(self):
        sft = sft_result(); sft.pop('adapter_reload')
        self.assertFalse(self.c.validate_result(manifest()['jobs'][0], sft, '1')['valid'])
        mace = mace_result(); mace['batch_equivalence']['per_batch_size']['4']['max_energy_difference_eV'] = 1e-5
        self.assertFalse(self.c.validate_result(manifest()['jobs'][1], mace, '2')['valid'])

    def test_missing_sft_counts_and_mace_relaxations_are_unusable(self):
        sft = sft_result(); sft['microbatches'] = 63
        self.assertFalse(self.c.validate_result(manifest()['jobs'][0], sft, '1')['valid'])
        mace = mace_result(); mace['relaxations'].pop()
        self.assertFalse(self.c.validate_result(manifest()['jobs'][1], mace, '2')['valid'])

    def test_state_completed_but_exit_nonzero_is_not_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp); out = home/'outputs/sft_1'; out.mkdir(parents=True)
            (out/'result.json').write_text(json.dumps(dict(status='completed', seen_tokens=100,
                action_tokens=20, optimizer_steps=32, wall_seconds=10,
                runtime={'job_id': '1', 'visible_gpus': 1})))
            record = dict(state='COMPLETED', exit_code='1:0', elapsed_seconds=20,
                          actual_card_hours=20/3600, verified=True)
            row = self.c.analyze_job(home, manifest()['jobs'][0], dict(slurm_job_id='1'), record)
            self.assertFalse(row['usable'])

    def test_terminal_retrieval_fetches_adapter_hashes_without_weights(self):
        import hashlib
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); home = root/'local'; remote = root/'remote'
            output = remote/'outputs/sft_1'; (output/'adapter').mkdir(parents=True)
            (output/'result.json').write_text('{"status":"failed"}')
            (output/'adapter/adapter_model.safetensors').write_bytes(b'example weights')
            (remote/'jobs/sft').mkdir(parents=True)
            (remote/'jobs/sft/slurm-1.out').write_text('retained failure log')
            spec = manifest(); spec['remote_smoke_root'] = str(remote)
            def fake_ssh(host, command, payload, timeout):
                return subprocess.run(['python3', '-'], input=payload, capture_output=True, check=True)
            with patch.object(self.c, 'ssh', fake_ssh):
                self.c.retrieve(home, spec, 'fake', {'jobs': [dict(id='c07_smoke_sft', slurm_job_id='1')]},
                                {'1': dict(state='FAILED')})
            local = home/'outputs/sft_1'
            meta = json.loads((local/'ADAPTER_METADATA.json').read_text())
            self.assertEqual(meta['adapter_files'][0]['sha256'], hashlib.sha256(b'example weights').hexdigest())
            self.assertFalse(list(local.rglob('*.safetensors')))
            self.assertEqual((local/'slurm-1.out').read_text(), 'retained failure log')

    def test_guard_preserves_corrupt_stop_and_filters_other_user_jobs(self):
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); home = root/'local'; remote = root/'remote'; remote.mkdir()
            (remote/'STOP_SMOKE.json').write_text('{broken original')
            binary = root/'bin'; binary.mkdir()
            (binary/'squeue').write_text('#!/bin/sh\nprintf "1|PENDING\\n2|RUNNING\\n777|PENDING\\n"\n')
            (binary/'scancel').write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> '+str(root/'cancelled')+'\n')
            for path in binary.iterdir():path.chmod(0o755)
            spec = manifest(); spec['remote_smoke_root'] = str(remote)
            ledger = {'jobs': [dict(id='c07_smoke_sft', slurm_job_id='1'), dict(id='c07_smoke_mace', slurm_job_id='2')]}
            plan = self.c.guard_plan([dict(id='c07_smoke_sft', slurm_job_id='1', state='PENDING'),
                                     dict(id='c07_smoke_mace', slurm_job_id='2', state='RUNNING')],
                                     prepared_budget(), ['allocation anomaly'], self.c.read_marker(remote/'STOP_SMOKE.json'))
            def fake_ssh(host, command, payload, timeout):
                return subprocess.run(['python3', '-'], input=payload, capture_output=True, check=True,
                    env=dict(os.environ, PATH=str(binary)+':'+os.environ['PATH']))
            with patch.object(self.c, 'ssh', fake_ssh):
                result = self.c.apply_guard(home, spec, 'fake', ledger, plan)
            self.assertEqual(result['pending_cancellation_requests'], ['1'])
            self.assertEqual((root/'cancelled').read_text().strip(), '--state=PENDING 1')
            self.assertEqual((remote/'STOP_SMOKE.json').read_text(), '{broken original')

    def test_offline_orchestration_exposes_parent_summary_and_keeps_failure_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            ledger = {'jobs': [dict(id='c07_smoke_sft', slurm_job_id='1'), dict(id='c07_smoke_mace', slurm_job_id='2')]}
            for name, value in [('BUDGET.json', prepared_budget()), ('JOB_MANIFEST.json', manifest()), ('SUBMISSIONS.json', ledger),
                    ('LATEST.json', dict(checked_at_utc='fixture', remote_ledger=ledger, stop=None,
                        account=dict(code=0, stdout='1|FAILED|450|cpu=6,gres/gpu=1|1:0|6|1\n2|PENDING|0||0:0|0|0\n'),
                        queue=dict(code=0, stdout='2|PENDING\n')))]:
                (home/name).write_text(json.dumps(value))
            analysis = self.c.collect(home, 'fake', offline=True)
            self.assertFalse(analysis['all_jobs_terminal'])
            self.assertFalse(analysis['all_terminal_outputs_retrieved'])
            self.assertFalse(analysis['guard_plan']['pool_stop_required'])
            self.assertAlmostEqual(json.loads((home/'BUDGET.json').read_text())['actual_card_hours'], .125)
            self.assertFalse((home/'STOP_SMOKE.json').exists())

    def test_remote_query_reads_and_reconciles_without_changing_remote_files(self):
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); home = root/'local'; home.mkdir(); remote = root/'remote'; remote.mkdir()
            ledger = {'jobs': [dict(id='c07_smoke_sft', slurm_job_id='1')]}
            (home/'SUBMISSIONS.json').write_text(json.dumps(ledger))
            (remote/'SUBMISSIONS.json').write_text(json.dumps(ledger))
            original = (remote/'SUBMISSIONS.json').read_bytes()
            binary = root/'bin'; binary.mkdir()
            (binary/'sacct').write_text('#!/bin/sh\nprintf "1|PENDING|0||0:0|0|0\\n"\n')
            (binary/'squeue').write_text('#!/bin/sh\nprintf "1|PENDING\\n"\n')
            for path in binary.iterdir():path.chmod(0o755)
            spec = manifest(); spec['remote_smoke_root'] = str(remote)
            def fake_ssh(host, command, payload, timeout):
                return subprocess.run(['python3', '-'], input=payload, capture_output=True, check=True,
                    env=dict(os.environ, PATH=str(binary)+':'+os.environ['PATH']))
            with patch.object(self.c, 'ssh', fake_ssh):
                snapshot = self.c.query(home, spec, 'fake')
            self.assertEqual(snapshot['ledger']['ledger_status'], 'RECONCILED')
            self.assertEqual((remote/'SUBMISSIONS.json').read_bytes(), original)
            self.assertEqual({p.name for p in remote.iterdir()}, {'SUBMISSIONS.json'})

    def test_transient_query_failure_defers_without_touching_budget_ledger_or_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            ledger = {'jobs': [dict(id='c07_smoke_sft', slurm_job_id='1'), dict(id='c07_smoke_mace', slurm_job_id='2')]}
            for name, value in [('BUDGET.json', prepared_budget()), ('JOB_MANIFEST.json', manifest()), ('SUBMISSIONS.json', ledger),
                    ('LATEST.json', dict(checked_at_utc='fixture', remote_ledger=ledger, stop=None,
                        account=dict(code=1, stdout='', stderr='temporary sacct service error'),
                        queue=dict(code=0, stdout='1|RUNNING\n2|PENDING\n')))]:
                (home/name).write_text(json.dumps(value))
            before = {name: (home/name).read_bytes() for name in ['BUDGET.json', 'SUBMISSIONS.json']}
            with patch.object(self.c, 'apply_guard', side_effect=AssertionError('Guard cannot run on an unknown query')):
                analysis = self.c.collect(home, 'fake', do_guard=True, offline=True)
            self.assertEqual(analysis['collection_status'], 'DEFERRED')
            self.assertFalse(analysis['guard_plan']['pool_stop_required'])
            for name, value in before.items():self.assertEqual((home/name).read_bytes(), value)
            self.assertFalse((home/'STOP_SMOKE.json').exists())

    def test_missing_running_accounting_is_deferred_not_a_consumption_anomaly(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            ledger = {'jobs': [dict(id='c07_smoke_sft', slurm_job_id='1')]}
            for name, value in [('BUDGET.json', prepared_budget()), ('JOB_MANIFEST.json', manifest()), ('SUBMISSIONS.json', ledger),
                    ('LATEST.json', dict(checked_at_utc='fixture', remote_ledger=ledger, stop=None,
                        account=dict(code=0, stdout=''), queue=dict(code=0, stdout='1|RUNNING\n')))]:
                (home/name).write_text(json.dumps(value))
            before = (home/'BUDGET.json').read_bytes()
            analysis = self.c.collect(home, 'fake', offline=True)
            self.assertEqual(analysis['collection_status'], 'DEFERRED')
            self.assertFalse(analysis['guard_plan']['pool_stop_required'])
            self.assertEqual((home/'BUDGET.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
