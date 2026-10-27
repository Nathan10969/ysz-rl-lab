"""CPU orchestration tests. Synthetic E/F and tiny policy are not GPU evidence."""
from dataclasses import asdict, replace
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import claude08_env as env
import smoke_grpo08 as old
from calibration import make_state
from claude10_physics import BatchPhysicsResult
try:
    grpo = importlib.import_module('smoke_grpo10')
except ModuleNotFoundError as exc:
    if exc.name != 'smoke_grpo10': raise
    grpo = None


class Actor:
    def __init__(self, events=None, illegal_lanes=(), invalid_gate=False, fail_at=None, varied=False):
        self.events = [] if events is None else events
        self.illegal_lanes, self.invalid_gate = set(illegal_lanes), invalid_gate
        self.fail_at, self.varied = fail_at, varied
        self.rollout_calls = self.reference_calls = self.updates = 0
        self.closed = False
        self.samples = []

    def generate(self, messages, ledger, *, phase, occupancy):
        ledger.before_prefill(3, phase=phase+'_generation')
        ledger.before_generated_token(phase=phase+'_sample')
        lane = self.rollout_calls % 4
        if phase == 'rollout':
            self.events.append(('generate', self.rollout_calls//4, lane))
            self.rollout_calls += 1
            if self.rollout_calls == self.fail_at: raise RuntimeError('synthetic generation failure')
        text = env.enumerate_legal_actions(occupancy)[0].to_json()
        if (phase == 'legality' and self.invalid_gate) or (phase == 'rollout' and lane in self.illegal_lanes):
            text = 'bad'
        # Fixed token fixture includes a prompt and an action/EOS target mask.
        action = old.GeneratedAction(text, (1,2,3,4), (False,False,True,True),
                                     (-1.,-1.1,-1.2), (-1.,-1.1,-1.2))
        self.samples.append(action)
        return action

    def score_reference(self, action, ledger):
        ledger.before_prefill(len(action.input_ids), phase='reference_full_context')
        self.events.append(('reference', self.reference_calls//4, self.reference_calls%4))
        self.reference_calls += 1
        return replace(action, reference_logprobs=(-1.5,-1.6,-1.7))

    def update(self, episodes, returns, ledger):
        self.updates += 1
        self.episodes, self.returns = episodes, returns
        return dict(optimizer_updates=1, finite_gradients=True, parameter_delta_l2=.01,
                    status='UPDATED', record_hashes=[[s.sha256 for s in e] for e in episodes])

    def finalize(self, ledger, output): return dict(save_unload_reload=True)
    def close(self): self.closed = True


class Physics:
    def __init__(self, events=None, *, fail_batch=None, unconverged_batch=None, lane_energy=False):
        self.events = [] if events is None else events
        self.fail_batch, self.unconverged_batch = fail_batch, unconverged_batch
        self.lane_energy = lane_energy
        self.batches = []
        self.last_evidence = []
        self.closed = False

    def relax_many(self, requests, counter, max_steps=300):
        assert threading.current_thread() is threading.main_thread(), 'Physics must run on main thread'
        assert max_steps == 300
        previous = sum(map(len, self.batches))
        assert counter.counters.relax_actual == previous + len(requests), 'Relaxation must be precharged'
        self.batches.append(list(requests))
        self.events.append(('batch', len(self.batches)-1, len(requests)))
        counter.before_forward(len(requests))
        self.last_evidence = [dict(formal_receipt=True,status='converged',raw_ef_attempts=1,
            nsteps=0,forward_calls=[counter.counters.forward_calls],lane=i) for i in range(len(requests))]
        if len(self.batches) == self.fail_batch: raise RuntimeError('synthetic charged physical failure')
        return [BatchPhysicsResult(r.atoms.copy(), 100.-(.01*i if self.lane_energy and previous else 0.),
                    np.zeros((len(r.atoms),3)), len(self.batches) != self.unconverged_batch,
                    evidence=self.last_evidence[i]) for i,r in enumerate(requests)]

    def close(self): self.closed = True


class GRPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.starts = [tuple(map(int,make_state(820+i,3,16).numbers)) for i in range(4)]
        cls.heldout = [dict(occupancy=cls.starts[i%4], messages=[{'role':'user','content':'cpu fixture'}]) for i in range(20)]
        # Memoize only identical ideal geometry. Distinct inputs use real mapping.
        cls.real_mapping, cls.mapping_cache = staticmethod(env.map_terminal), {}

    def setUp(self):
        self.assertIsNotNone(grpo, 'Review10 synchronized runner is required')
        def mapping(atoms):
            key = (atoms.numbers.tobytes(),atoms.positions.tobytes(),atoms.cell.array.tobytes(),atoms.pbc.tobytes())
            if key not in self.mapping_cache: self.mapping_cache[key] = self.real_mapping(atoms)
            return self.mapping_cache[key]
        self.mapping_patch = patch.object(env,'map_terminal',side_effect=mapping)
        self.mapping_patch.start()
        self.addCleanup(self.mapping_patch.stop)

    def run_fake(self, actor=None, physics=None, **kwargs):
        actor = Actor() if actor is None else actor
        physics = Physics() if physics is None else physics
        return grpo.run_smoke(actor, physics, self.starts, self.heldout, **kwargs)

    def test_failed_twenty_gate_never_scores_reference_or_calls_physics(self):
        actor, physics = Actor(invalid_gate=True), Physics()
        result = self.run_fake(actor,physics)
        self.assertEqual(result['status'],'NOT_RELEASED_LEGALITY_GATE')
        self.assertEqual((result['legality']['denominator'],len(result['legality']['generations'])),(20,20))
        self.assertEqual((physics.batches, actor.reference_calls, actor.updates),([],0,0))
        self.assertTrue(actor.closed and physics.closed)
        self.assertEqual(result['counters']['generated_tokens'],20)

    def test_four_actions_are_synchronized_and_unknown_duplicates_are_four_real_lanes(self):
        events = []
        actor, physics = Actor(events), Physics(events)
        result = self.run_fake(actor,physics)
        self.assertEqual(result['status'],'NO_UPDATE_ZERO_GROUP_VARIANCE')
        self.assertEqual(result['optimizer_updates'],0)
        self.assertEqual(result['counters']['attempts'],128)
        first_rollout_batch = events.index(('batch',1,4))
        self.assertEqual([e for e in events[:first_rollout_batch] if e[0]=='generate'],
                         [('generate',0,i) for i in range(4)])
        self.assertEqual([e for e in events[:first_rollout_batch] if e[0]=='reference'],
                         [('reference',0,i) for i in range(4)])
        self.assertEqual(len(set(r.key for r in physics.batches[1])),1)
        self.assertEqual(len(physics.batches[1]),4)
        hits = [s['outcome']['cache_hit'] for s in result['groups'][0]['episodes'][0]['steps']]
        self.assertFalse(hits[0])
        self.assertTrue(any(hits[1:]))
        # Each private rollout can discover the same family before group merge.
        self.assertEqual([e['steps'][0]['outcome']['novelty_reward'] for e in result['groups'][0]['episodes']], [.05]*4)
        self.assertEqual(result['counters']['relax_actual'],sum(map(len,physics.batches)))
        self.assertEqual(result['counters']['raw_ef'],sum(map(len,physics.batches)))

    def test_illegal_lanes_produce_real_three_lane_batch_and_one_joint_update(self):
        actor, physics = Actor(illegal_lanes={3}), Physics()
        result = self.run_fake(actor,physics)
        self.assertEqual(result['status'],'COMPLETED_ONE_UPDATE')
        self.assertEqual(actor.updates,1)
        self.assertEqual(len(physics.batches[1]),3)
        self.assertEqual(len(actor.episodes),16)
        self.assertTrue(all(len(e)==8 for e in actor.episodes))
        sample = result['groups'][0]['episodes'][0]['steps'][0]['sample']
        self.assertEqual(tuple(sample['old_logprobs']),(-1.,-1.1,-1.2))
        self.assertEqual(tuple(sample['reference_logprobs']),(-1.5,-1.6,-1.7))
        self.assertEqual(tuple(sample['action_mask']),(False,False,True,True))
        self.assertEqual(actor.samples[20].old_logprobs,(-1.,-1.1,-1.2))

    def test_two_and_one_lane_batches_are_not_padded(self):
        for lanes, expected in [({2,3},2),({1,2,3},1)]:
            physics = Physics()
            result = self.run_fake(Actor(illegal_lanes=lanes),physics)
            self.assertEqual(len(physics.batches[1]),expected)
            self.assertEqual(result['optimizer_updates'],1)

    def test_failed_forward_keeps_all_charges_generated_actions_and_partial_group(self):
        actor, physics = Actor(), Physics(fail_batch=2)
        result = self.run_fake(actor,physics)
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual((result['counters']['relax_actual'],result['counters']['raw_ef']),(8,8))
        self.assertEqual(actor.rollout_calls,4)
        self.assertEqual(actor.updates,0)
        self.assertEqual(len(result['partial_episodes']),4)
        self.assertTrue(all(len(e['generated'])==1 for e in result['partial_episodes']))
        self.assertEqual(result['physical_batches'][-1]['status'],'failed')
        self.assertFalse(result['usable_for_workflow'])

    def test_unconverged_physics_stops_without_next_sampling_or_update(self):
        actor, physics = Actor(), Physics(unconverged_batch=2)
        result = self.run_fake(actor,physics)
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual(actor.rollout_calls,4)
        self.assertEqual((actor.updates,len(physics.batches)),(0,2))
        self.assertEqual(result['counters']['raw_ef'],8)
        lanes = result['physical_batches'][-1]['lanes']
        self.assertTrue(all('result' in lane and 'forces_eV_A' in lane['result'] for lane in lanes))

    def test_shared_relax_cap_keeps_exact_precharged_partial_batch_and_stops(self):
        actor, physics = Actor(), Physics()
        result = self.run_fake(actor,physics,limits=replace(old.LIMITS,max_relax_actual=6))
        self.assertEqual(result['status'],'BUDGET_EXHAUSTED')
        self.assertEqual([len(b) for b in physics.batches],[4,2])
        self.assertEqual((result['counters']['relax_actual'],result['counters']['raw_ef']),(6,6))
        self.assertEqual(actor.rollout_calls,4)
        self.assertEqual(actor.updates,0)

    def test_raw_cap_prevents_overlarge_forward_and_retains_relax_attempts(self):
        physics = Physics()
        result = self.run_fake(physics=physics,limits=replace(old.LIMITS,max_raw_ef=6))
        self.assertEqual(result['status'],'BUDGET_EXHAUSTED')
        self.assertEqual(result['counters']['raw_ef'],4)
        self.assertEqual(result['counters']['relax_actual'],8)
        self.assertEqual(result['optimizer_updates'],0)

    def test_generation_failure_preserves_earlier_actions_without_physical_batch(self):
        actor, physics = Actor(fail_at=3), Physics()
        result = self.run_fake(actor,physics)
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual([len(b) for b in physics.batches],[4])
        self.assertEqual(sum(len(e['generated']) for e in result['partial_episodes']),2)
        self.assertEqual(result['counters']['generated_tokens'],23)

    def test_worker_failure_is_joined_and_waiting_peers_are_resolved(self):
        actual_step = env.Environment.step
        calls = []
        def broken(environment,*args,**kwargs):
            calls.append(environment)
            if len(calls)==2: raise RuntimeError('synthetic worker failure')
            return actual_step(environment,*args,**kwargs)
        begin = time.monotonic()
        with patch.object(env.Environment,'step',broken):
            result = self.run_fake()
        self.assertLess(time.monotonic()-begin,5.)
        self.assertEqual(result['status'],'FAILED')
        self.assertFalse(any(t.name.startswith('review10-step-') for t in threading.enumerate()))

    def test_wall_cap_during_physical_batch_resolves_all_waiters_and_stops(self):
        now = [0.]
        class ExpiringPhysics(Physics):
            def relax_many(self,*args,**kwargs):
                values = super().relax_many(*args,**kwargs)
                if len(self.batches)==2: now[0]=6600.
                return values
        actor = Actor()
        result = self.run_fake(actor,ExpiringPhysics(),clock=lambda:now[0],started_at=0.)
        self.assertEqual(result['status'],'BUDGET_EXHAUSTED')
        self.assertEqual(actor.rollout_calls,4)
        self.assertFalse(any(t.name.startswith('review10-step-') for t in threading.enumerate()))

    def test_final_write_overrun_is_durable_and_unusable(self):
        now = [0.]
        actual_save = old.save_json
        def save(path,value):
            actual_save(path,value)
            if Path(path).name=='result.json': now[0]=6601.
        with tempfile.TemporaryDirectory() as tmp, patch.object(grpo,'save_json',side_effect=save):
            output = Path(tmp)/'run'
            result = self.run_fake(Actor(invalid_gate=True),output=output,clock=lambda:now[0],started_at=0.)
            self.assertEqual(result['status'],'BUDGET_EXHAUSTED')
            self.assertFalse(result['usable_for_workflow'])
            self.assertEqual(json.loads((output/'result.json').read_text())['status'],'BUDGET_EXHAUSTED')

    def test_real_joint_math_preserves_action_records_and_changes_parameter_once(self):
        class TinyActor(Actor):
            def __init__(self):
                super().__init__(illegal_lanes={3})
                self.weight = torch.nn.Parameter(torch.tensor(-1.))
            def update(self,episodes,returns,ledger):
                self.updates += 1
                def score(sample):
                    ledger.before_prefill(len(sample.input_ids),phase='current_training_full_context')
                    return self.weight.expand(3)
                return old.one_update([self.weight],episodes,returns,score,before_step=ledger.check)
        actor = TinyActor()
        result = self.run_fake(actor)
        self.assertEqual(result['status'],'COMPLETED_ONE_UPDATE')
        self.assertEqual(actor.updates,1)
        self.assertTrue(result['update']['finite_gradients'])
        self.assertGreater(result['update']['parameter_delta_l2'],0.)


class ReleaseTests(unittest.TestCase):
    """Schema units isolate teacher/data's separately tested semantic recovery."""
    def setUp(self):
        self.assertIsNotNone(grpo)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.occupancy = tuple(map(int,make_state(920,3,16).numbers))
        self.text = env.enumerate_legal_actions(self.occupancy)[0].to_json()
        self.rows = [dict(occupancy=self.occupancy) for _ in range(80)]
        import claude10_data as data
        self.data_sources = data.source_hashes()
        self.data_patch = patch.object(data,'load_frozen_data',side_effect=lambda path:dict(
            manifest_sha256=old.sha256(Path(path)/'manifest.json'),
            manifest=json.loads((Path(path)/'manifest.json').read_text()),
            rows={'legality_sft.jsonl':self.rows}))
        self.data_patch.start(); self.addCleanup(self.data_patch.stop)
        self.teacher_patch = patch.object(data,'verify_teacher_result',return_value=[{}]*16)
        self.teacher_verifier = self.teacher_patch.start(); self.addCleanup(self.teacher_patch.stop)
        self.manifest = self.fixture()

    def write(self,name,value):
        path = self.root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,allow_nan=False))
        return {'path':name,'sha256':old.sha256(path)}

    def fixture(self):
        data_manifest = self.write('data/manifest.json',{'schema':'unit-fixture'})
        qwen = self.write('qwen.json',{'repository':'Qwen/Qwen3-0.6B','status':'verified_complete',
            'files':[{'path':'model.safetensors','sha256':old.QWEN_WEIGHT_SHA}]})
        checkpoint = self.root/'checkpoint.model'; checkpoint.write_bytes(b'unit fixture frozen MACE')
        self.model_patch = patch.object(env,'MODEL_SHA',old.sha256(checkpoint))
        self.model_patch.start(); self.addCleanup(self.model_patch.stop)
        reload = dict(unloaded_without_merge=True,reloaded_from_disk=True,state_exact_match=True)
        config = self.write('adapter/adapter_config.json',dict(r=16,lora_alpha=32,lora_dropout=.05,
                                                            target_modules=['q_proj','v_proj']))
        adapter = self.write('adapter/adapter_model.safetensors',{'synthetic':'adapter bytes'})
        legality = dict(status='passed',denominator=80,attempts=80,legal=76,records=[dict(index=i,
            status='completed',text=self.text if i<76 else 'bad',legal=i<76) for i in range(80)])
        import smoke_sft10 as sft
        sft_result = dict(schema='review10-sft-v1',status='completed',seed=old.SEED,config=sft.CONFIG,
            adapter_release_allowed=True,GPU_work_performed=True,physical_calls=0,
            source_hashes=self.data_sources,runtime={'slurm_job_id':'653'},training_runs=1,validation_retries=0,
            data_manifest_sha256=data_manifest['sha256'],qwen_manifest_sha256=qwen['sha256'],
            training=dict(parameters_changed=True,optimizer_updates=1,adapter_hash_before='a',adapter_hash_after='b',
                          updates=[dict(gradient_norm=.1)]),reload=reload,legality=legality)
        sft_proof = dict(independently_verified=True,finite_gradients=True,parameters_changed=True,
            adapter_reload=reload,adapter_files=[config,adapter],legality={k:legality[k] for k in ('denominator','attempts','legal')})
        teacher_result = dict(schema='review10-teacher-v1',status='completed',seed=old.SEED,diagnostic=False,
            runtime={'slurm_job_id':'652','GPU_work_performed':True},source_hashes=self.data_sources,
            counters={'relax_actual':240,'raw_ef':240,'forward_calls':100})
        teacher_proof = dict(independently_verified=True,job_id='652',formal_batch_bfgs_verified=True,
            all_240_receipts_verified=True,complete_trajectories=16,attempts_per_trajectory=8,
            raw_ef=240,relax_actual=240,forward_calls=100,parent_exit_code='0:0')
        teacher_ref = self.write('data/teacher_result.json',teacher_result)
        data_manifest = self.write('data/manifest.json',{'files':{'teacher_result.json':teacher_ref['sha256']}})
        sft_result['data_manifest_sha256'] = data_manifest['sha256']
        sft_result['reload'] = dict(reload,adapter_sha256=adapter['sha256'],adapter_config_sha256=config['sha256'])
        parity = {'4':{'passed':True},'16':{'passed':True}}
        mace_result = dict(status='completed',runtime={'job_id':'648'},raw_ef_calls=1023,
            checkpoint_sha256=env.MODEL_SHA,batch_equivalence={'per_batch_size':parity})
        mace_proof = dict(independently_verified=True,job_id='648',raw_ef=1023,singleton_batch_parity=parity,
                          retry_of='635',authorized_replacement_index=1)
        import claude10_physics as physics
        cpu_result = dict(status='cpu_execution_preflight_passed',production_code_path=True,
            runtime={'device':'cpu','GPU_work_performed':False},source_hashes=physics.source_hashes(),
            runner_source_hashes=grpo.source_hashes(),checkpoint_sha256=env.MODEL_SHA,
            short_receipts_for_training=False,parity_tolerance=1e-6,
            parity=[dict(lane=i,batch_single_energy_difference_eV=0.,batch_single_force_difference_eV_A=0.,
                    batch_native_energy_difference_eV=0.,batch_native_force_difference_eV_A=0.) for i in range(4)],
            relaxations={str(size):[dict(nsteps=1,raw_ef_attempts=2,execution_max_steps=1,
                                       formal_receipt=False) for _ in range(size)] for size in (1,4)},
            actual_forward_audit=dict(observed_raw_ef=22,observed_forward_calls=13,all_precharged=True,matches_counter=True),
            counters={'raw_ef':22,'forward_calls':13})
        cpu_proof = dict(independently_verified=True,CPU_only=True,same_production_code_path=True,batch_sizes=[1,4])
        root_proof = dict(independently_verified=True,accounting_complete=True,accounting_unknown_job_ids=[],
            all_prior_jobs_terminal=True,review10_grpo_submissions=0,
            review10_actual_plus_remaining_caps_including_this=6.,root_actual_plus_remaining_caps_including_this=7.)
        def stage(name,job,result,proof):
            return dict(job_id=job,terminal='COMPLETED',recovered=True,
                        result=self.write(name+'_result.json',result),verification=self.write(name+'_verify.json',proof))
        return dict(schema='review10-release-v1',seed=old.SEED,source_hashes=grpo.source_hashes(),
            authorization=dict(review=10,retry_of='650',authorized_retry_index=1,max_card_hours=2,additional_card_hours=6),
            data_manifest=data_manifest,qwen_manifest=qwen,
            checkpoint={'path':'checkpoint.model','sha256':old.sha256(checkpoint)},
            sft=stage('sft','653',sft_result,sft_proof),teacher652=stage('teacher','652',teacher_result,teacher_proof),
            mace648=stage('mace','648',mace_result,mace_proof),
            cpu_preflight=dict(result=self.write('cpu_result.json',cpu_result),verification=self.write('cpu_verify.json',cpu_proof)),
            root_fees=dict(verification=self.write('fees_verify.json',root_proof)))

    def check(self):
        self.write('release.json',self.manifest)
        return grpo.verify_release_manifest(self.root/'release.json')

    def mutate(self,section,kind,fn):
        ref = self.manifest[section][kind]
        value = json.loads((self.root/ref['path']).read_text())
        fn(value)
        self.manifest[section][kind] = self.write(ref['path'],value)

    def test_release_binds_new_sft_adapter_and_independent_semantic_teacher_recovery(self):
        checked = self.check()
        self.assertTrue(checked['verified'])
        old.verify_adapter_binding(self.root/'adapter',checked)
        self.teacher_verifier.assert_called_once()
        self.assertEqual(checked['data_manifest_sha256'],self.manifest['data_manifest']['sha256'])

    def test_recomputed_legality_rejects_forged_seventy_six_count(self):
        self.mutate('sft','result',lambda r:r['legality']['records'][75].update(text='bad'))
        with self.assertRaisesRegex(ValueError,'legality'): self.check()

    def test_sft_below_gate_missing_reload_and_nonfinite_gradients_rejected(self):
        initial = json.loads(json.dumps(self.manifest))
        for alter in [lambda r:r['legality'].update(legal=75),
                      lambda r:r['reload'].update(state_exact_match=False),
                      lambda r:r['training']['updates'][0].update(gradient_norm=0.),
                      lambda r:r['training'].update(parameters_changed=False)]:
            self.manifest = json.loads(json.dumps(initial))
            # Restore the original source result before applying each mutation.
            result = json.loads((self.root/'sft_result.json').read_text())
            result['legality'].update(legal=76)
            result['reload'].update(state_exact_match=True)
            result['training'].update(parameters_changed=True,updates=[dict(gradient_norm=.1)])
            alter(result)
            self.manifest['sft']['result'] = self.write('sft_result.json',result)
            with self.assertRaises(ValueError): self.check()

    def test_changed_adapter_source_hash_unknown_fees_or_second_retry_rejected(self):
        checked = self.check()
        (self.root/'adapter/adapter_model.safetensors').write_text('tampered')
        with self.assertRaises(ValueError): self.check()
        with self.assertRaises(ValueError): old.verify_adapter_binding(self.root/'adapter',checked)
        # Remaining independent checks use a restored hash-bound synthetic adapter.
        ref = self.write('adapter/adapter_model.safetensors',{'synthetic':'adapter bytes'})
        self.manifest['authorization']['authorized_retry_index'] = 2
        with self.assertRaisesRegex(ValueError,'authorization'): self.check()
        self.manifest['authorization']['authorized_retry_index'] = 1
        self.mutate('root_fees','verification',lambda r:r.update(accounting_unknown_job_ids=['unknown']))
        with self.assertRaisesRegex(ValueError,'accounting'): self.check()
        self.mutate('root_fees','verification',lambda r:r.update(accounting_unknown_job_ids=[]))
        self.manifest['source_hashes']['smoke_grpo10.py'] = '0'*64
        with self.assertRaisesRegex(ValueError,'source'): self.check()

    def test_cpu_preflight_requires_new_runner_binding_and_real_native_bfgs_evidence(self):
        self.mutate('cpu_preflight','result',lambda r:r['runner_source_hashes'].update(smoke_grpo10_py='fake'))
        with self.assertRaisesRegex(ValueError,'source'): self.check()

    def test_adapter_reload_and_frozen_teacher_bytes_cannot_be_swapped(self):
        self.mutate('sft','result',lambda r:r['reload'].update(adapter_sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'adapter'): self.check()
        self.mutate('sft','result',lambda r:r['reload'].update(
            adapter_sha256=old.sha256(self.root/'adapter/adapter_model.safetensors')))
        self.mutate('teacher652','result',lambda r:r.update(unrelated_recovered_result=True))
        with self.assertRaisesRegex(ValueError,'teacher'): self.check()

    def test_short_bfgs_claim_without_actual_one_step_and_missing_native_parity_rejected(self):
        self.mutate('cpu_preflight','result',lambda r:r['relaxations']['4'][0].update(nsteps=0))
        with self.assertRaisesRegex(ValueError,'BFGS'): self.check()
        self.mutate('cpu_preflight','result',lambda r:r['relaxations']['4'][0].update(nsteps=1))
        self.mutate('cpu_preflight','result',lambda r:r['parity'][0].pop('batch_native_energy_difference_eV'))
        with self.assertRaisesRegex(ValueError,'parity'): self.check()

    def test_cli_rejects_release_before_guard_actor_or_physics_initialization(self):
        self.write('release.json',dict(self.manifest,source_hashes={}))
        with patch.object(old,'require_single_gpu_job') as guard, patch.object(grpo,'HFActor') as actor, \
             patch.object(grpo,'BatchMACEPhysics') as physics:
            with self.assertRaises(ValueError):
                grpo.main(['--model',str(self.root/'base'),'--adapter',str(self.root/'adapter'),
                    '--qwen-manifest',str(self.root/'qwen.json'),'--checkpoint',str(self.root/'checkpoint.model'),
                    '--release-manifest',str(self.root/'release.json'),'--data',str(self.root/'data'),
                    '--output',str(self.root/'never_created')])
            guard.assert_not_called(); actor.assert_not_called(); physics.assert_not_called()
        self.assertFalse((self.root/'never_created').exists())

    def test_cpu_preflight_wrapper_calls_same_backend_and_rejects_existing_output(self):
        from claude10_physics import cpu_execution_preflight
        def fake(checkpoint,output):
            output.mkdir()
            return dict(status='cpu_execution_preflight_passed')
        with patch('claude10_physics.cpu_execution_preflight',side_effect=fake) as backend:
            output = self.root/'explicit_cpu'
            result = grpo.real_cpu_preflight(self.root/'checkpoint.model',output)
            backend.assert_called_once()
            self.assertEqual(result['runner_source_hashes'],grpo.source_hashes())
            self.assertTrue((output/'result.json').exists())
            with self.assertRaises(FileExistsError): grpo.real_cpu_preflight(self.root/'checkpoint.model',output)


if __name__ == '__main__': unittest.main()
