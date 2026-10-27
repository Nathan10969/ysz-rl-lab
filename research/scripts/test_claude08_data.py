"""Synthetic CPU tests: no teacher physics or data freeze is executed in the repo."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from calibration import make_state
import claude08_env as env
try:
    data = importlib.import_module('claude08_data')
except ModuleNotFoundError as exc:
    if exc.name != 'claude08_data': raise
    data = None

class Tokenizer:
    eos_token_id = 255
    def apply_chat_template(self, messages, **kw):
        if kw['enable_thinking'] is not False: raise ValueError('Thinking must be disabled')
        return 'P:' + ''.join(m['content'] for m in messages) + ':A'
    def __call__(self, text, **kw): return {'input_ids': list(text.encode())}

class DataTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(data, 'Review08 CPU data implementation is required')
    def test_action_only_rows_and_no_truncation(self):
        row = {'messages':[{'role':'user','content':'hello'}], 'action':{'op':'swap_cations','sites':[0,1]}}
        encoded = data.action_sft_row(Tokenizer(), row)
        n = encoded['prompt_tokens']
        self.assertEqual(encoded['labels'][:n], [-100]*n)
        self.assertEqual(encoded['labels'][n:], encoded['input_ids'][n:])
        self.assertEqual(encoded['input_ids'][-1], 255)
        self.assertEqual(encoded['action_mask'], [False]*n+[True]*(len(encoded['input_ids'])-n))
        with self.assertRaises(ValueError): data.action_sft_row(Tokenizer(), row, max_tokens=8)
    def test_streams_deterministic_distinct_and_family_split_has_no_overlap(self):
        self.assertEqual(data.seed_streams(42), data.seed_streams(42))
        self.assertEqual(len(set(data.seed_streams(42))), 3)
        rows = [{'terminal_families':['a','b'], 'id':0}, {'terminal_families':['b','c'], 'id':1},
                {'terminal_families':['d'], 'id':2}, {'terminal_families':['e'], 'id':3}]
        train, valid = data.split_terminal_families(rows, seed=8, validation_fraction=.5)
        tf = set().union(*(set(r['terminal_families']) for r in train))
        vf = set().union(*(set(r['terminal_families']) for r in valid))
        self.assertFalse(tf & vf)
        self.assertEqual(len(train)+len(valid), 4)
    def test_synthetic_teacher_claim_is_rejected(self):
        with self.assertRaises(ValueError): data.validate_teacher_trajectory({'purpose':'unscored synthetic legal action'})
    def test_verified_receipt_requires_convergence_and_positive_recorded_cost(self):
        occupancy = tuple(map(int, make_state(90,3,16).numbers))
        request = env.build_request(occupancy)
        result = env.PhysicsResult(request.atoms, 3., np.zeros((316,3)), True)
        receipt = data.PhysicalReceipt.verify(request, result, relax_actual=1, raw_ef=2)
        self.assertEqual(receipt.terminal_key, env.occupancy_key(occupancy))
        self.assertEqual(receipt.costs['raw_ef'], 2)
        result.converged = False
        with self.assertRaises(ValueError): data.PhysicalReceipt.verify(request,result,relax_actual=1,raw_ef=2)
        result.converged = True
        with self.assertRaises(ValueError): data.PhysicalReceipt.verify(request,result,relax_actual=0,raw_ef=0)
    def test_freeze_is_callable_deterministic_disjoint_and_hashes_all_files(self):
        ideals = [tuple(map(int, make_state(i,3,16).numbers)) for i in range(8)]
        forbidden = {env.occupancy_key(occ) for occ in ideals}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'future_freeze'
            with self.assertRaises(ValueError): data.freeze_smoke_data(path, ideals, forbidden, prerequisites=None)
            self.assertFalse(path.exists())
            manifest = data.freeze_smoke_data(path, ideals, forbidden, prerequisites={'verified':True})
            starts = [json.loads(x) for x in (path/'starts.jsonl').read_text().splitlines()]
            heldout = [json.loads(x) for x in (path/'legality_v2.jsonl').read_text().splitlines()]
            self.assertEqual((len(starts),len(heldout)), (4,20))
            keys = [r['canonical_key'] for r in starts+heldout]
            self.assertEqual(len(set(keys)),24)
            self.assertTrue({r['canonical_key'] for r in starts} <= forbidden)
            self.assertFalse({r['canonical_key'] for r in heldout}&forbidden)
            self.assertTrue(all(r['unscored'] is True for r in heldout))
            self.assertTrue(all('candidate' not in json.dumps(r['messages']) for r in heldout))
            self.assertTrue(all(data.sha256(path/name)==digest for name,digest in manifest['files'].items()))

    def test_receipt_metadata_is_frozen_and_reverified(self):
        request=env.build_request(tuple(map(int,make_state(91,3,16).numbers)))
        result=env.PhysicsResult(request.atoms,3.,np.zeros((316,3)),True)
        receipt=data.PhysicalReceipt.verify(request,result,relax_actual=1,raw_ef=2)
        with self.assertRaises(TypeError): receipt.costs['raw_ef']=500
        for field,value in [('model_sha','wrong'),('converged',False),('energy',4.)]:
            changed=data.PhysicalReceipt.verify(request,result,relax_actual=1,raw_ef=2)
            object.__setattr__(changed,field,value)
            with self.assertRaises(ValueError): changed.reverify()

    def test_stale_teacher_messages_are_rebuilt_from_verified_previous_terminal(self):
        occupancy=tuple(map(int,make_state(92,3,16).numbers))
        request=env.build_request(occupancy)
        initial=data.PhysicalReceipt.verify(request,env.PhysicsResult(request.atoms,100.,np.zeros((316,3)),True),relax_actual=1,raw_ef=2)
        action=env.enumerate_legal_actions(occupancy)[0]
        next_request=env.build_request(env.apply_action(occupancy,action))
        receipt=data.PhysicalReceipt.verify(next_request,env.PhysicsResult(next_request.atoms,99.9,np.zeros((316,3)),True),relax_actual=1,raw_ef=3)
        stale=[{'role':'user','content':'an unrelated stale structure'}]
        trajectory={'complete':True,'initial_receipt':initial,'transitions':[{'messages':stale,'action':json.loads(action.to_json()),'receipt':receipt}]}
        validated=data.validate_teacher_trajectory(trajectory)
        messages=validated['rows'][0]['messages']
        self.assertNotIn('unrelated stale',json.dumps(messages))
        observation=json.loads(messages[-1]['content'])
        self.assertEqual(env.decode_occupancy(observation['occupancy']),occupancy)
        self.assertEqual(observation['energy_eV'],100.)
        self.assertEqual(observation['counts']['raw_ef'],2)
        self.assertEqual(observation['remaining_attempts'],8)
        def callback(next_request,counter):
            counter.before_forward(1)
            return env.PhysicsResult(next_request.atoms,99.9,np.zeros((316,3)),True)
        collected=data.collect_teacher(initial,lambda *args:(stale,action.to_json()),callback,max_steps=1,
            counters=env.Counters(relax_actual=1,raw_ef=2),limits=env.BudgetLimits(max_relax_actual=3,max_raw_ef=10),seed=42)
        self.assertEqual(collected['transitions'][0]['messages'],data.validate_teacher_trajectory(collected)['rows'][0]['messages'])
        self.assertNotEqual(collected['transitions'][0]['messages'],stale)

if __name__ == '__main__': unittest.main()
