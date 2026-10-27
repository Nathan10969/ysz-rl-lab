"""CPU fake tokenizer/tiny network; no transformer weights or GPU operations."""
import importlib
from pathlib import Path
import sys
import json
import tempfile
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import claude08_env as env
from calibration import make_state
try:
    sft = importlib.import_module('smoke_sft10')
except ModuleNotFoundError as exc:
    if exc.name != 'smoke_sft10': raise
    sft = None


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(8))
    def forward(self, input_ids, attention_mask, labels, **kwargs):
        logits = self.logits.expand(*input_ids.shape, 8)
        return type('Output', (), {'loss':torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 8), labels[:, 1:].reshape(-1), ignore_index=-100)})()


class SFTTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(sft, 'Review10 SFT implementation is required')

    def test_padding_masks_labels_and_partial_accumulation_updates(self):
        samples = [{'input_ids':[1,2,3], 'labels':[-100,-100,3], 'prompt_tokens':2,'action_tokens':1},
                   {'input_ids':[1,4], 'labels':[-100,4], 'prompt_tokens':1,'action_tokens':1}] * 5
        batch = sft.collate_rows(samples[:2], pad_token_id=0, device='cpu')
        self.assertEqual(batch['attention_mask'].tolist(), [[1,1,1],[1,1,0]])
        self.assertEqual(batch['labels'].tolist(), [[-100,-100,3],[-100,4,-100]])
        plan = sft.build_schedule(samples)
        self.assertEqual(plan['microbatches'], 6)
        self.assertEqual(plan['optimizer_updates'], 4)
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        result = sft.train_updates(model, optimizer, samples, plan, sft.RuntimeGuard(),
                                   pad_token_id=0, device='cpu')
        self.assertEqual(result['optimizer_updates'], 4)
        self.assertTrue(result['parameters_changed'])
        self.assertTrue(all(r['gradient_norm'] > 0 for r in result['updates']))
        self.assertEqual(result['seen_padded_tokens'], plan['seen_padded_tokens'])

    def test_legality_gate_uses_eighty_once_without_resampling(self):
        occupancy = tuple(map(int, make_state(713, 3, 16).numbers))
        action = env.enumerate_legal_actions(occupancy)[0].to_json()
        rows = [{'occupancy':occupancy,'messages':[]} for _ in range(80)]
        for legal_count, expected in [(76,'passed'), (75,'NOT_RELEASED_SFT_LEGALITY_GATE')]:
            calls = []
            def generate(messages):
                calls.append(messages)
                return action if len(calls) <= legal_count else '<|im_start|>' + action
            result = sft.evaluate_legality(rows, generate, sft.RuntimeGuard())
            self.assertEqual(len(calls), 80)
            self.assertEqual(result['denominator'], 80)
            self.assertEqual(result['legal'], legal_count)
            self.assertEqual(result['status'], expected)

    def test_runtime_guard_fails_at_cap_and_preserves_special_tokens(self):
        guard = sft.RuntimeGuard(started_at=0., clock=lambda:6600.)
        with self.assertRaises(env.BudgetExceeded): guard.check()
        class Tokenizer:
            eos_token_id = 3
            def decode(self, tokens, *, skip_special_tokens):
                self.tokens = tokens
                self.skip = skip_special_tokens
                return '<special>{}'
        tokenizer = Tokenizer()
        self.assertEqual(sft.decode_action_tokens(tokenizer, [1,2,3]), '<special>{}')
        self.assertEqual(tokenizer.tokens, [1,2])
        self.assertFalse(tokenizer.skip)

    def test_generation_samples_every_vocabulary_token_and_retains_control_token(self):
        class Tokenizer:
            eos_token_id = 2
            def apply_chat_template(self, messages, **kw): return [0]
            def decode(self, tokens, *, skip_special_tokens):
                self.tokens, self.skip = tokens, skip_special_tokens
                return '<control>'
        class Generator:
            def __call__(self, **kw):
                return type('Output', (), {'logits':torch.tensor([[[0.,1.,2.,3.]]]),
                                           'past_key_values':None})()
        probabilities = []
        def multinomial(probs, n, generator):
            probabilities.append(probs.tolist())
            return torch.tensor([1 if len(probabilities)==1 else 2])
        counts = {'prefill_tokens':0,'generated_tokens':0,'seen_tokens':0}
        tokenizer = Tokenizer()
        with patch('torch.multinomial', side_effect=multinomial):
            text = sft.sample_action(Generator(),tokenizer,[],sft.RuntimeGuard(),None,counts,device='cpu')
        self.assertEqual(text,'<control>')
        self.assertEqual(tokenizer.tokens,[1])
        self.assertFalse(tokenizer.skip)
        self.assertTrue(all(p>0 for p in probabilities[0]))
        self.assertTrue(torch.allclose(torch.tensor(probabilities[0]),torch.tensor([0.,1.,2.,3.]).softmax(-1)))
        self.assertEqual(counts, {'prefill_tokens':1,'generated_tokens':2,'seen_tokens':3})

    def test_third_generation_failure_preserves_completed_texts_and_charged_attempt(self):
        occupancy = tuple(map(int, make_state(715, 3, 16).numbers))
        text = env.enumerate_legal_actions(occupancy)[0].to_json()
        rows = [{'occupancy':occupancy,'messages':[]} for _ in range(80)]
        counts = {'prefill_tokens':0,'generated_tokens':0,'seen_tokens':0}
        stage = {'status':'running','legality':{},'generation_tokens':counts}
        calls = []
        def generate(messages):
            calls.append(messages)
            counts['prefill_tokens'] += 10
            counts['seen_tokens'] += 10
            generated = 2 if len(calls)==3 else 1
            counts['generated_tokens'] += generated
            counts['seen_tokens'] += generated
            if len(calls)==3:
                stage['legality']['active_attempt']['text'] = '<control>partial'
                raise RuntimeError('synthetic generation failure')
            return text
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp)/'legality_attempts.jsonl'
            try:
                sft.evaluate_legality(rows,generate,sft.RuntimeGuard(),progress=stage['legality'],
                                      generation_counts=counts,journal=journal)
            except RuntimeError as exc:
                self.assertEqual(str(exc),'synthetic generation failure')
                stage['status'] = 'failed'
            else:
                self.fail('The third generation must fail without retry')
            sft.persist_result(Path(tmp)/'result.json',stage,started_at=0.,clock=lambda:1.)
            saved = json.loads((Path(tmp)/'result.json').read_text())
            events = [json.loads(line) for line in journal.read_text().splitlines()]
        self.assertEqual(len(calls),3)
        self.assertEqual(saved['status'],'failed')
        self.assertEqual(saved['legality']['status'],'failed')
        self.assertEqual(saved['legality']['denominator'],80)
        self.assertEqual(saved['legality']['attempts'],3)
        self.assertEqual([r['text'] for r in saved['legality']['records']],[text,text])
        self.assertEqual(saved['generation_tokens'],{'prefill_tokens':30,'generated_tokens':4,'seen_tokens':34})
        self.assertEqual(saved['legality']['failed_attempt']['text'],'<control>partial')
        self.assertEqual(saved['legality']['failed_attempt']['token_delta'],
                         {'prefill_tokens':10,'generated_tokens':2,'seen_tokens':12})
        self.assertEqual([e['event'] for e in events],['started','completed','started','completed','started','failed'])


if __name__ == '__main__': unittest.main()
