"""CPU diagnostics only: injected energies are synthetic, never teacher evidence."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import claude08_env as env
from calibration import make_state
try:
    teacher = importlib.import_module('smoke_teacher10')
except ModuleNotFoundError as exc:
    if exc.name != 'smoke_teacher10': raise
    teacher = None


class FakePhysics:
    def __init__(self, energies, fail=False, unconverged=False):
        self.energies = iter(energies)
        self.requests = []
        self.fail, self.unconverged = fail, unconverged
        self.last_evidence = []

    def relax_many(self, requests, counter, max_steps=300):
        self.requests.extend(requests)
        counter.before_forward(len(requests))
        self.last_evidence = [dict(raw_ef_attempts=1, nsteps=0, formal_receipt=True,
                                   status='converged',forward_calls=[counter.counters.forward_calls]) for _ in requests]
        if self.fail: raise RuntimeError('synthetic charged failure')
        return [SimpleResult(r.atoms.copy(), next(self.energies), np.zeros((316, 3)),
                             not self.unconverged, evidence=self.last_evidence[i])
                for i, r in enumerate(requests)]


class SimpleResult(env.PhysicsResult):
    def __init__(self, *args, evidence):
        super().__init__(*args)
        self.evidence = evidence


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(teacher, 'Review10 teacher implementation is required')

    def initial(self):
        return env.build_request(tuple(map(int, make_state(710, 3, 16).numbers)))

    def test_sa_rejections_keep_current_terminal_and_still_finish_eight_attempts(self):
        physics = FakePhysics([0.] + [1000.] * 8)
        counter = env.RawEFCounter(env.Counters(), env.BudgetLimits(max_relax_actual=240, max_raw_ef=72240))
        records = []
        initial = teacher.evaluate_batch(physics, [self.initial()], counter, records)[0]
        trajectory = teacher.run_trajectory('SA', initial, physics, counter, records,
                                            rng=np.random.default_rng(81), trajectory_id=0)
        self.assertEqual(len(trajectory['steps']), 8)
        self.assertTrue(all(s['accepted_id'] is None for s in trajectory['steps']))
        self.assertTrue(all(s['before_id'] == initial['id'] for s in trajectory['steps']))
        self.assertEqual(counter.counters.relax_actual, 9)
        self.assertEqual(counter.counters.raw_ef, 9)
        self.assertEqual(len(records), 9)
        self.assertTrue(all(r['accepted'] is False for r in records[1:]))

    def test_ga_uses_measured_fitness_and_breeds_legal_next_population(self):
        physics = FakePhysics([50.] + [9., -5., 2., 7.] * 8)
        counter = env.RawEFCounter(env.Counters(), env.BudgetLimits(max_relax_actual=240, max_raw_ef=72240))
        records = []
        initial = teacher.evaluate_batch(physics, [self.initial()], counter, records)[0]
        trajectory = teacher.run_trajectory('GA', initial, physics, counter, records,
                                            rng=np.random.default_rng(82), trajectory_id=0)
        for step in trajectory['steps']:
            candidates = [records[i] for i in step['candidate_ids']]
            self.assertEqual(step['accepted_id'], candidates[1]['id'])
            self.assertEqual(step['selected_parent_ids'], [candidates[1]['id'], candidates[2]['id']])
            before = records[step['accepted_id']]['receipt']['terminal_occupancy']
            for child in step['breeding']:
                env.apply_action(before, env.parse_action(json.dumps(child['action'])))
                self.assertIn(child['parent_ids'][0], step['selected_parent_ids'])
                self.assertIn('crossover_sites', child)
                self.assertIn('mutated', child)
        self.assertEqual(counter.counters.relax_actual, 33)

    def test_cli_accepts_only_empty_backend_directory_and_never_overwrites_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'teacher'
            (output/'physics').mkdir(parents=True)
            teacher.prepare_output(output)
            (output/'result.json').write_text('{}')
            with self.assertRaises(FileExistsError): teacher.prepare_output(output)

    def test_failed_collection_closes_backend_inside_stage_and_keeps_durable_failure(self):
        physics = FakePhysics([],fail=True)
        closed = []
        physics.close = lambda:closed.append(True)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'diagnostic'
            with self.assertRaisesRegex(RuntimeError,'synthetic charged failure'):
                teacher.collect_teacher(physics,output=output)
            self.assertEqual(closed,[True])
            saved = json.loads((output/'result.json').read_text())
            self.assertEqual(saved['status'],'failed')
            self.assertEqual(saved['cleanup']['status'],'completed')
            self.assertFalse(saved['runtime']['GPU_work_performed'])

    def test_production_guard_metadata_is_bound_into_durable_result(self):
        # Emulate only the production type/guard boundary; this test executes no GPU work.
        physics = FakePhysics([],fail=True)
        physics.device = 'cuda'
        runtime = {'slurm_job_id':'synthetic-test-only','cuda_visible_devices':'0','device_count':1,
                   'device_name':'synthetic-no-GPU'}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'diagnostic'
            with patch('claude10_physics.BatchMACEPhysics',FakePhysics), \
                 patch('smoke_grpo08.require_single_gpu_job',return_value=runtime):
                with self.assertRaisesRegex(RuntimeError,'synthetic charged failure'):
                    teacher.collect_teacher(physics,output=output)
            saved = json.loads((output/'result.json').read_text())
            for key,value in runtime.items(): self.assertEqual(saved['runtime'][key],value)
            self.assertEqual(saved['status'],'failed')

    def test_physics_failure_keeps_precharged_cost_and_raw_failure_record(self):
        for physics in [FakePhysics([], fail=True), FakePhysics([0.], unconverged=True)]:
            counter = env.RawEFCounter(env.Counters(), env.BudgetLimits(max_relax_actual=240, max_raw_ef=72240))
            records = []
            with self.assertRaises((RuntimeError, ValueError)):
                teacher.evaluate_batch(physics, [self.initial()], counter, records)
            self.assertEqual(counter.counters.relax_actual, 1)
            self.assertEqual(counter.counters.raw_ef, 1)
            self.assertEqual(records[0]['status'], 'failed')
            self.assertEqual(records[0]['physics_evidence']['raw_ef_attempts'], 1)
            if physics.unconverged: self.assertIn('result', records[0])

    def test_full_fixed_collection_recovers_all_connections_and_rejects_tampering(self):
        import claude10_data as data
        physics = FakePhysics([0.] * 240)
        # Mapping is a pure geometry function. Cache only identical readonly inputs;
        # each distinct geometry still passes the real assignment implementation.
        actual_mapping, cache = env.map_terminal, {}
        def mapping(atoms):
            key = (atoms.numbers.tobytes(),atoms.positions.tobytes(),atoms.cell.array.tobytes(),atoms.pbc.tobytes())
            if key not in cache: cache[key] = actual_mapping(atoms)
            return cache[key]
        with patch.object(env,'map_terminal',side_effect=mapping):
            result = teacher.collect_teacher(physics)
            self.assertTrue(result['diagnostic'])
            self.assertFalse(result['runtime']['GPU_work_performed'])
            self.assertEqual(result['counters']['relax_actual'], 240)
            self.assertEqual(result['counters']['attempts'], 128)
            recovered = data.verify_teacher_result(result)
            self.assertEqual(sum(len(t['rows']) for t in recovered), 128)
            ga_second = recovered[6]['rows'][1]
            counts = json.loads(ga_second['messages'][-1]['content'])['counts']
            # Initial lane + first four-lane GA batch: five structures, two actual forwards.
            self.assertEqual(counts['raw_ef'],5)
            self.assertEqual(counts['forward_calls'],2)
            first = result['trajectories'][0]['steps'][0]
            result['records'][first['candidate_ids'][0]]['before_id'] = -1
            with self.assertRaisesRegex(ValueError, 'connect'):
                data.verify_teacher_result(result)


if __name__ == '__main__': unittest.main()
