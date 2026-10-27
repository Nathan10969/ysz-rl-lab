"""CPU injection tests for the fixed Review08 smoke protocol."""
import importlib
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import torch
import unittest
import numpy as np
from calibration import make_state
import claude08_env as env
try:
    smoke = importlib.import_module('smoke_grpo08')
except ModuleNotFoundError as exc:
    if exc.name != 'smoke_grpo08': raise
    smoke = None

class FakeActor:
    def __init__(self, invalid_gate=False, varied=False):
        self.invalid_gate=invalid_gate; self.varied=varied; self.updates=0; self.closed=False
    def generate(self, messages, ledger, *, phase, occupancy):
        ledger.before_prefill(3, phase=phase+'_generation')
        ledger.before_generated_token(phase=phase)
        action = 'bad' if (phase=='legality' and self.invalid_gate) else env.enumerate_legal_actions(occupancy)[0].to_json()
        if phase!='legality' and self.varied and ledger.counters.attempts%32==0:
            action = 'bad'
        return smoke.GeneratedAction(action, (1,2,3,4), (False,False,False,True), (-1.,-1.,-1.), (-1.,-1.,-1.))
    def score_reference(self, action, ledger): return action
    def update(self, episodes, returns, ledger):
        self.updates+=1
        return {'optimizer_updates':1,'finite_gradients':True,'parameter_delta_l2':.01,'status':'UPDATED'}
    def finalize(self, ledger, output): return {'save_unload_reload':True}
    def close(self): self.closed=True

class SmokeTests(unittest.TestCase):
    def test_CPU_preflight_does_not_change_production_BFGS_limits(self):
        production=smoke.MACEPhysics('not-loaded',None,'.')
        self.assertEqual(getattr(production,'device',None),'cuda')
        self.assertEqual(production.max_steps,300)
        cpu=smoke.MACEPhysics('not-loaded',None,'.',cpu_preflight=True)
        self.assertEqual((cpu.device,cpu.max_steps),('cpu',1))

    @classmethod
    def setUpClass(cls):
        cls.starts=[tuple(map(int,make_state(200+i,3,16).numbers)) for i in range(4)]
        cls.heldout=[{'occupancy':tuple(map(int,make_state(300+i,3,16).numbers)), 'messages':[{'role':'user','content':'state'}]} for i in range(20)]
    def setUp(self): self.assertIsNotNone(smoke, 'Review08 smoke implementation is required')
    def physics(self, requests):
        def callback(request,counter):
            self.assertGreater(counter.counters.relax_actual, len(requests))
            requests.append(request.key); counter.before_forward(1)
            return env.PhysicsResult(request.atoms, 100.,np.zeros((316,3)),True)
        return callback
    def test_legality_gate_failure_has_no_physics_or_update_and_closes(self):
        actor=FakeActor(invalid_gate=True); requests=[]
        result=smoke.run_smoke(actor,self.physics(requests),self.starts,self.heldout)
        self.assertEqual(result['status'],'NOT_RELEASED_LEGALITY_GATE')
        self.assertEqual(result['legality']['denominator'],20)
        self.assertEqual((requests,actor.updates),([],0)); self.assertTrue(actor.closed)
        self.assertGreater(result['counters']['generated_tokens'],0)
    def test_initial_charge_is_four_and_zero_variance_does_not_update(self):
        actor=FakeActor(); requests=[]
        result=smoke.run_smoke(actor,self.physics(requests),self.starts,self.heldout)
        self.assertEqual(result['initial_relaxations'],4)
        self.assertEqual(len(result['groups']),4)
        self.assertEqual(result['counters']['attempts'],128)
        self.assertEqual(result['status'],'NO_UPDATE_ZERO_GROUP_VARIANCE')
        self.assertEqual(actor.updates,0)
        self.assertLessEqual(result['counters']['relax_actual'],132)
        self.assertEqual(result['counters']['generated_tokens'],148)
        self.assertEqual(result['counters']['prefill_tokens'],444)
    def test_shared_cap_stops_incomplete_group_without_pg(self):
        actor=FakeActor(); requests=[]
        limits=env.BudgetLimits(max_attempts=8,max_relax_actual=132,max_raw_ef=50000,
            max_generated_tokens=23,max_prefill_tokens=293632,max_seen_tokens=262144,wall_seconds=6600)
        result=smoke.run_smoke(actor,self.physics(requests),self.starts,self.heldout,limits=limits)
        self.assertEqual(result['status'],'BUDGET_EXHAUSTED')
        self.assertEqual(result['groups'],[]); self.assertEqual(actor.updates,0)
        self.assertEqual(result['counters']['generated_tokens'],23)
    def test_before_call_caps_and_runtime_guard(self):
        counters=env.Counters(); ledger=smoke.TokenLedger(counters,env.BudgetLimits(max_generated_tokens=1,max_prefill_tokens=4,max_seen_tokens=4))
        ledger.before_prefill(3,phase='test'); ledger.before_generated_token(phase='test')
        with self.assertRaises(env.BudgetExceeded): ledger.before_generated_token(phase='test')
        self.assertEqual(counters.seen_tokens,4)
        with self.assertRaises(RuntimeError): smoke.require_single_gpu_job(environ={}, torch_module=None)
    def test_varied_groups_get_at_most_one_joint_update(self):
        actor=FakeActor(varied=True); requests=[]
        result=smoke.run_smoke(actor,self.physics(requests),self.starts,self.heldout)
        self.assertEqual(actor.updates,1)
        self.assertEqual(result['groups'][0]['episodes'][0]['illegal_return'],-1.)
        self.assertEqual(result['groups'][0]['episodes'][0]['physical_failure_return'],0.)
        self.assertEqual(result['update']['optimizer_updates'],1)
        self.assertTrue(result['adapter_check']['save_unload_reload'])
        self.assertTrue(actor.closed)

    def test_hashed_release_rejects_dependency_only_and_tampered_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def write(name,value):
                path=root/name; path.write_text(json.dumps(value))
                return {'path':name,'sha256':smoke.sha256(path)}
            reload={'unloaded_without_merge':True,'reloaded_from_disk':True,'state_exact_match':True}
            adapter=write('adapter_model.safetensors',{'fake':'synthetic file for checksum only'})
            config=write('adapter_config.json',{'fake':'synthetic config for checksum only'})
            sft=write('633_result.json',{'status':'completed','adapter_reload':reload,'optimizer_steps':32})
            sv=write('633_verify.json',{'independently_verified':True,'finite_gradients':True,'adapter_reload':reload,'adapter_files':[adapter,config]})
            parity={'4':{'passed':True},'16':{'passed':True}}
            mace=write('635_result.json',{'status':'completed','raw_ef_calls':456,'batch_equivalence':{'per_batch_size':parity}})
            mv=write('635_verify.json',{'independently_verified':True,'raw_ef':456,'singleton_batch_parity':parity})
            release={'schema':'review08-release-v1',
                'sft633':{'job_id':633,'terminal':'COMPLETED','recovered':True,'result':sft,'verification':sv},
                'mace635':{'job_id':635,'terminal':'COMPLETED','recovered':True,'result':mace,'verification':mv}}
            manifest=root/'release.json';manifest.write_text(json.dumps(release))
            checked=smoke.verify_release_manifest(manifest)
            self.assertTrue(checked['verified'])
            self.assertEqual(checked['adapter_hashes']['adapter_model.safetensors'],adapter['sha256'])
            smoke.verify_adapter_binding(root,checked)
            replacement=json.loads(json.dumps(release))
            replacement['schema']='review08-release-v2'
            replacement['mace635'].update(job_id='648',retry_of='635',authorized_replacement_index=1)
            replacement['mace635']['result']=write('648_result.json',{'status':'completed','raw_ef_calls':456,
                'runtime':{'job_id':'648'},'batch_equivalence':{'per_batch_size':parity}})
            replacement['mace635']['verification']=write('648_verify.json',{'independently_verified':True,
                'retry_of':'635','authorized_replacement_index':1,'raw_ef':456,'singleton_batch_parity':parity})
            replacement['grpo_cpu_preflight']=write('cpu.json',{'independently_verified':True,
                'CPU_only':True,'same_production_code_path':True,'batch_sizes':[1,4],
                'source_sha256':{name:smoke.sha256(Path(smoke.__file__).parent/name)
                    for name in ['smoke_grpo08.py','smoke_mace.py']}})
            manifest.write_text(json.dumps(replacement))
            try:
                self.assertTrue(smoke.verify_release_manifest(manifest)['verified'])
            except ValueError as error:
                self.fail('The explicitly authorized635 replacement and real CPU receipt must be accepted: '+str(error))
            replacement['mace635']['authorized_replacement_index']=2
            manifest.write_text(json.dumps(replacement))
            with self.assertRaises(ValueError): smoke.verify_release_manifest(manifest)
            manifest.write_text(json.dumps(release))
            (root/'adapter_model.safetensors').write_text('changed')
            with self.assertRaises(ValueError): smoke.verify_release_manifest(manifest)
            with self.assertRaises(ValueError): smoke.verify_adapter_binding(root,checked)
            release['sft633']={'dependency':'afterok:633'};manifest.write_text(json.dumps(release))
            with self.assertRaises(ValueError): smoke.verify_release_manifest(manifest)

    def test_raw_hook_charges_before_actual_torch_forward_and_cache_is_not_counted(self):
        class Model(torch.nn.Module):
            def forward(self,data):
                self.asserted=counters.raw_ef
                if data.get('fail'): raise RuntimeError('synthetic forward failed')
                return torch.ones(1)
        model=Model();counters=env.Counters();limits=env.BudgetLimits(max_raw_ef=2)
        counter=env.RawEFCounter(counters,limits)
        request=env.build_request(self.starts[0])
        with tempfile.TemporaryDirectory() as tmp:
            hook=smoke.attach_raw_ef_hook(model,lambda:(counter,request),Path(tmp)/'attempts.jsonl')
            model({'batch':torch.zeros(316,dtype=torch.long)})
            self.assertEqual(model.asserted,1)
            with self.assertRaises(RuntimeError): model({'batch':torch.zeros(316,dtype=torch.long),'fail':True})
            self.assertEqual(counters.raw_ef,2)
            with self.assertRaises(env.BudgetExceeded): model({'batch':torch.zeros(316,dtype=torch.long)})
            self.assertEqual(len((Path(tmp)/'attempts.jsonl').read_text().splitlines()),2)
            hook.remove()

    def test_actual_torch_update_through_complete_orchestration(self):
        from claude08_grpo import one_update
        class TorchActor(FakeActor):
            def __init__(self):
                super().__init__(varied=True);self.weight=torch.nn.Parameter(torch.tensor(-1.))
            def generate(self,*args,**kwargs):
                action=super().generate(*args,**kwargs)
                token=1 if action.text=='bad' else 2
                return smoke.replace(action,input_ids=(1,2,3,token))
            def score_reference(self,action,ledger):
                ledger.before_prefill(len(action.input_ids),phase='fake_reference')
                return action
            def update(self,episodes,returns,ledger):
                self.updates+=1
                def score(sample):
                    ledger.before_prefill(len(sample.input_ids),phase='fake_current_training')
                    return (self.weight if sample.input_ids[-1]==1 else self.weight*.5-.5).expand(3)
                return one_update([self.weight],episodes,returns,score,before_step=ledger.check)
        actor=TorchActor();requests=[]
        with tempfile.TemporaryDirectory() as tmp:
            result=smoke.run_smoke(actor,self.physics(requests),self.starts,self.heldout,output=tmp)
            saved=json.loads((Path(tmp)/'result.json').read_text())
            receipts=saved['physics_receipts']
            initial=[r for r in receipts if r['phase']=='initial']
            self.assertEqual(len(initial),4)
            for record in receipts:
                receipt=json.loads((Path(tmp)/record['receipt_file']).read_text())
                self.assertEqual(receipt['status'],'verified_converged')
                self.assertEqual(len(receipt['terminal']['forces_eV_A']),316)
                self.assertEqual(len(receipt['terminal']['positions_A']),316)
                self.assertEqual(len(receipt['terminal']['mapped_occupancy']),324)
                self.assertGreater(receipt['raw_ef_end'],receipt['raw_ef_begin'])
                self.assertEqual(receipt['protocol'],env.PROTOCOL)
            self.assertIn('gpu_peak_allocated_bytes',saved['resources'])
            self.assertIn('gpu_peak_reserved_bytes',saved['resources'])
            self.assertIn('torch',saved['runtime_versions'])
            self.assertTrue(all(row['seconds']>=0 for row in saved['phase_timings']))
            self.assertTrue(any(row['phase']=='policy_update' for row in saved['phase_timings']))
        self.assertEqual(result['status'],'COMPLETED_ONE_UPDATE')
        self.assertGreater(result['update']['parameter_delta_l2'],0)
        self.assertTrue(result['update']['finite_gradients'])
        self.assertEqual(result['counters'],saved['counters'])
        self.assertEqual(len(result['update']['record_hashes']),16)
        self.assertEqual(result['counters']['prefill_tokens'],444+128*4*2)

    def test_failed_initial_physics_keeps_attempt_receipt_and_raw_cost(self):
        def failed(request,counter):
            counter.before_forward(1)
            raise RuntimeError('synthetic physical failure')
        with tempfile.TemporaryDirectory() as tmp:
            result=smoke.run_smoke(FakeActor(),failed,self.starts,self.heldout,output=tmp)
            self.assertEqual(result['status'],'FAILED')
            self.assertEqual(result['counters']['raw_ef'],1)
            self.assertEqual(result['counters']['relax_actual'],1)
            self.assertEqual(len(result['physics_receipts']),1)
            row=json.loads((Path(tmp)/result['physics_receipts'][0]['receipt_file']).read_text())
            self.assertEqual(row['status'],'failed')
            self.assertEqual(row['raw_ef_end'],1)
            self.assertNotIn('terminal',row)
            self.assertIn('synthetic physical failure',row['error'])

    def test_failed_reload_cannot_be_marked_usable(self):
        actor=FakeActor()
        actor.finalize=lambda *args:{'save_unload_reload':False}
        result=smoke.run_smoke(actor,self.physics([]),self.starts,self.heldout)
        self.assertEqual(result['status'],'FAILED')
        self.assertFalse(result['usable_for_workflow'])

    def test_decode_preserves_every_special_token_except_final_eos(self):
        class Tokenizer:
            eos_token_id=3
            def decode(self,ids,*,skip_special_tokens):
                self.skip=skip_special_tokens
                return ''.join({1:'<control>',2:'{"op":"move_vacancy","sites":[1,2]}',3:'<eos>'}[i] for i in ids)
        tokenizer=Tokenizer()
        text=smoke.decode_action_tokens(tokenizer,[1,2,3])
        self.assertFalse(tokenizer.skip)
        self.assertTrue(text.startswith('<control>'))
        self.assertNotIn('<eos>',text)
        with self.assertRaises(env.ActionError): env.parse_action(text)
        self.assertEqual(env.parse_action(smoke.decode_action_tokens(tokenizer,[2,3])).op,'move_vacancy')
        self.assertIn('<eos>',smoke.decode_action_tokens(tokenizer,[3,2]))

    def test_final_persistence_overrun_is_unusable_and_included_in_elapsed(self):
        class Clock:
            value=0.
            def __call__(self): return self.value
        clock=Clock();original=smoke.save_json
        def slow_save(path,value):
            original(path,value)
            if Path(path).name=='result.json': clock.value+=2.
        limits=smoke.replace(smoke.LIMITS,wall_seconds=1.)
        with tempfile.TemporaryDirectory() as tmp,patch.object(smoke,'save_json',side_effect=slow_save):
            result=smoke.run_smoke(FakeActor(),self.physics([]),self.starts,self.heldout,
                                   output=tmp,limits=limits,clock=clock)
            saved=json.loads((Path(tmp)/'result.json').read_text())
        self.assertFalse(result['usable_for_workflow']);self.assertFalse(saved['usable_for_workflow'])
        self.assertEqual(result['status'],'BUDGET_EXHAUSTED')
        self.assertEqual(saved['status'],'BUDGET_EXHAUSTED')
        self.assertEqual(result['wall_seconds'],clock.value)
        self.assertGreaterEqual(result['wall_seconds'],2.)

    def test_physical_failure_return_is_logged_separately_from_illegal(self):
        requests=[]
        def physics(request,counter):
            requests.append(request.key);counter.before_forward(1)
            if len(requests)==5: raise RuntimeError('one synthetic physical failure')
            return env.PhysicsResult(request.atoms,100.,np.zeros((316,3)),True)
        result=smoke.run_smoke(FakeActor(),physics,self.starts,self.heldout)
        first=result['groups'][0]['episodes'][0]
        self.assertEqual(first['illegal_return'],0.)
        self.assertEqual(first['physical_failure_return'],-1.)
        self.assertEqual(first['invalid_return'],-1.)
        self.assertAlmostEqual(first['return'],first['energy_return']+first['novelty_return']+first['invalid_return'])

if __name__ == '__main__': unittest.main()
