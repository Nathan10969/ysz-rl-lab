"""Review12 LLM actor and SFT on top of the Review11 constrained actor.

Reference adapter = the seed's D0-SFT checkpoint (fixed). Policy adapter =
the arm's previous-round checkpoint (or the SFT checkpoint in round 1). Both
are loaded and then overwritten with the exact saved FP32 tensors (PEFT
0.17.1 rounds a second adapter through BF16 on load_adapter).
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

from claude08_data import action_sft_row
from claude08_grpo import state_fingerprint
from formal12_common import sha256, save_json, Phase
from smoke_grpo08 import QWEN_WEIGHT_SHA
import smoke_grpo11 as g11
from constrained import OLD_REFERENCE_TOLERANCE, ConstraintError, masked_causal_logprobs, one_update_masked

LORA = dict(r=16, lora_alpha=32, lora_dropout=.05, target_modules=['q_proj', 'v_proj'], bias='none', task_type='CAUSAL_LM')
SFT_CONFIG = dict(learning_rate=1e-4, minibatch=4, gradient_accumulation=2, clip_norm=1., max_tokens=2048)


def sft_row(tokenizer, row, *, max_tokens):
    """action rows (v2/v3: op/sites target) or menu rows (v4: {"choice":k} target); prompt+target+EOS."""
    if 'choice' not in row: return action_sft_row(tokenizer, row, max_tokens=max_tokens)
    from formal12_menu import choice_text
    prefix = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt = list(tokenizer(prefix, add_special_tokens=False)['input_ids'])
    full = list(tokenizer(prefix + choice_text(row['choice']), add_special_tokens=False)['input_ids'])
    if not prompt or full[:len(prompt)] != prompt: raise ValueError('Unstable prompt/choice token boundary')
    full.append(tokenizer.eos_token_id)
    if len(full) > max_tokens: raise ValueError('SFT context exceeds fixed cap')
    return {'input_ids': full, 'labels': [-100] * len(prompt) + full[len(prompt):],
            'action_mask': [False] * len(prompt) + [True] * (len(full) - len(prompt)), 'prompt_tokens': len(prompt),
            'action_tokens': len(full) - len(prompt)}


def adapter_hashes(path):
    return {name: sha256(Path(path) / name) for name in ('adapter_config.json', 'adapter_model.safetensors')}


def verify_qwen(model_path, manifest_path):
    provenance = json.loads(Path(manifest_path).read_text())
    if provenance.get('repository') != 'Qwen/Qwen3-0.6B' or provenance.get('status') != 'verified_complete':
        raise ValueError('Only verified official Qwen3-0.6B is approved')
    if not any(f['path'] == 'model.safetensors' and f['sha256'] == QWEN_WEIGHT_SHA for f in provenance['files']):
        raise ValueError('Qwen base weight SHA differs from approved manifest')
    for file in provenance['files']:
        if sha256(Path(model_path) / file['path']) != file['sha256']:
            raise ValueError('Qwen file hash changed: ' + file['path'])


class FormalActor(g11.ConstrainedHFActor):
    def __init__(self, model_path, reference_adapter, policy_adapter, qwen_manifest, *, output, device='cuda', dtype=None):
        release = dict(verified=True, adapter_hashes=adapter_hashes(reference_adapter))
        super().__init__(model_path, reference_adapter, qwen_manifest, release=release, output=output,
                         device=device, dtype=dtype)
        self.policy_adapter = Path(policy_adapter)
        self.policy_hashes_at_load = adapter_hashes(policy_adapter)
        self.updates = []
        self.grpo_lr = 1e-5                       # Review12.3: set per job from the tuning-selected config
        self.persistent_optimizer = False         # Review12.3 meta-layer option
        self._optimizer = None
        self.sft_lr = SFT_CONFIG['learning_rate']

    def _load(self, ledger):
        ledger.check()
        if self.run_device == 'cuda':
            from smoke_grpo08 import require_single_gpu_job
            require_single_gpu_job()
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        from safetensors.torch import load_file
        verify_qwen(self.model_path, self.qwen_manifest)
        for path in (self.adapter_path, self.policy_adapter):
            config = json.loads((path / 'adapter_config.json').read_text())
            if (config.get('r') != 16 or config.get('lora_alpha') != 32
                    or set(config.get('target_modules', [])) != {'q_proj', 'v_proj'}):
                raise ValueError('Adapter is not the fixed r16 alpha32 q/v LoRA: ' + str(path))
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=False)
        # MACE sets the global default dtype to float64; PEFT would then build
        # LoRA layers in float64. Load under float32 and restore afterwards.
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            base = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=False,
                                                        dtype=getattr(torch, self.dtype_name), attn_implementation='sdpa')
            self.model = PeftModel.from_pretrained(base, self.adapter_path, adapter_name='reference',
                                                   is_trainable=False, local_files_only=True)
            self.model.load_adapter(self.policy_adapter, adapter_name='policy', is_trainable=True, local_files_only=True)
        finally:
            torch.set_default_dtype(previous)
        named = dict(self.model.named_parameters())
        if any(p.dtype != torch.float32 for n, p in named.items() if 'lora_' in n):
            raise RuntimeError('LoRA adapter parameters must be float32')
        differences = {}
        with torch.no_grad():
            for label, path in (('reference', self.adapter_path), ('policy', self.policy_adapter)):
                saved = load_file(str(path / 'adapter_model.safetensors'))
                params = {n: n.replace('.' + label + '.', '.') for n in named if '.' + label + '.' in n}
                if len(params) != len(saved) or set(params.values()) != set(saved):
                    raise RuntimeError(label + ' adapter parameter set differs from file')
                differences[label] = max(float((named[n].float() - saved[k].float()).abs().max()) for n, k in params.items())
                for n, k in params.items(): named[n].copy_(saved[k].to(named[n].dtype))
        self.load_corrections = differences
        self._switch('policy')
        pairs = [(n, n.replace('.policy.', '.reference.')) for n in named if '.policy.' in n]
        self.identical_adapters_at_load = all(torch.equal(named[n], named[o]) for n, o in pairs)
        self.policy_load_max_abs_difference = differences['policy']
        self.initial_reference_hash = self._reference_hash()
        self.initial_policy_hash = self._policy_hash()
        self.fixed_hash = state_fingerprint({n: p for n, p in self.model.named_parameters() if '.policy.' not in n})
        self.to_device(self.run_device, ledger)
        self.generator = torch.Generator(device=self.run_device).manual_seed(int(self.seed_for_generator))

    seed_for_generator = 2026102301
    menu = None   # Review12.3 step 6: when set, generation chooses {"choice":k} over this menu

    def generate(self, messages, ledger, *, phase, occupancy):
        if self.menu is None:
            return super().generate(messages, ledger, phase=phase, occupancy=occupancy)
        import math as _math
        from formal12_menu import MenuTrie, choice_text
        from smoke_grpo08 import MAX_NEW_TOKENS, MAX_PROMPT_TOKENS, MAX_TOTAL_TOKENS, decode_action_tokens
        from constrained import SAMPLING_RECOMPUTE_STOP
        self._ensure(ledger); self._switch('policy')
        trie = MenuTrie(self.tokenizer, self.menu, max_new_tokens=MAX_NEW_TOKENS)
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if hasattr(ids, 'tolist'): ids = ids.tolist()
        if not ids or len(ids) > MAX_PROMPT_TOKENS: raise ValueError('Prompt exceeds 1984; no silent truncation')
        torch = self.torch
        inputs = torch.tensor([ids], device=self.run_device)
        ledger.before_prefill(len(ids), phase=phase + '_generation_prefill')
        generated, sampled_lp, allowed_rows, allowed_mass, argmax_allowed = [], [], [], [], []
        with torch.no_grad():
            output = self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=True)
            cache = output.past_key_values
            for index in range(MAX_NEW_TOKENS):
                allowed = trie.allowed(generated)
                scores = output.logits[0, -1].float()
                if not torch.isfinite(scores).all(): raise FloatingPointError('Nonfinite generation logits')
                index_tensor = torch.tensor(allowed, device=self.run_device)
                masked = scores.index_select(0, index_tensor).log_softmax(-1)
                allowed_mass.append(float(scores.log_softmax(-1).index_select(0, index_tensor).exp().sum()))
                argmax_allowed.append(int(scores.argmax()) in allowed)
                ledger.before_generated_token(phase=phase + '_sample')
                choice = 0 if len(allowed) == 1 else int(torch.multinomial(masked.exp(), 1, generator=self.generator).item())
                token = allowed[choice]
                generated.append(token); sampled_lp.append(float(masked[choice])); allowed_rows.append(allowed)
                if token == trie.eos: break
                if index + 1 >= MAX_NEW_TOKENS: raise ConstraintError('No EOS within the generation cap')
                ledger.check()
                attention = torch.ones((1, len(ids) + len(generated)), device=self.run_device, dtype=torch.long)
                output = self.model(input_ids=torch.tensor([[token]], device=self.run_device), attention_mask=attention,
                                    past_key_values=cache, use_cache=True)
                cache = output.past_key_values
        picked = trie.action(generated)
        text = decode_action_tokens(self.tokenizer, generated)
        if text != choice_text(picked['k']): raise ConstraintError('Decoded text differs from the menu choice text')
        full = tuple(ids + generated)
        if len(full) > MAX_TOTAL_TOKENS: raise ValueError('Total sequence exceeds 2048')
        aligned = tuple([()] * (len(ids) - 1) + allowed_rows)
        ledger.before_prefill(len(full), phase=phase + '_policy_full_context')
        logits, tensor = self._forward(full)
        lp = masked_causal_logprobs(logits, tensor, aligned).cpu().tolist()
        del logits
        gap = max(abs(a - b) for a, b in zip(lp[len(ids) - 1:], sampled_lp))
        self.records.append(dict(phase=phase, text=text, prompt_tokens=len(ids), action_tokens=len(generated),
            branch_tokens=sum(len(r) > 1 for r in allowed_rows), allowed_sizes=[len(r) for r in allowed_rows],
            allowed_probability_mass=allowed_mass, unconstrained_argmax_allowed=argmax_allowed, legal_set_size=len(trie),
            trie_sha256=trie.digest, sampling_vs_recompute_max_abs=gap, menu_choice=picked['k'],
            menu_dNN=picked['dNN'], menu_min_dNN=min(m['dNN'] for m in self.menu)))
        if not _math.isfinite(gap) or gap > SAMPLING_RECOMPUTE_STOP:
            raise ConstraintError('KV-cached sampling and full-context old log-probs disagree: ' + str(gap))
        mask = tuple([False] * len(ids) + [True] * len(generated))
        return g11.ConstrainedGeneratedAction(text, full, mask, tuple(lp), tuple(lp), aligned)

    def score_reference(self, action, ledger):
        """As Review11, but the zero-gap check applies only while policy == reference."""
        self._ensure(ledger); self._switch('reference')
        try:
            ledger.before_prefill(len(action.input_ids), phase='reference_full_context')
            logits, tensor = self._forward(action.input_ids)
            lp = masked_causal_logprobs(logits, tensor, action.allowed).cpu().tolist()
            del logits
            if self._reference_hash() != self.initial_reference_hash: raise RuntimeError('Reference adapter changed')
            branch = [t for t, row in enumerate(action.allowed) if len(row) > 1]
            gap = max(abs(lp[t] - action.old_logprobs[t]) for t in branch)
            if self.records and self.records[-1]['text'] == action.text:
                self.records[-1]['old_vs_reference_max_abs'] = gap
            if self.identical_adapters_at_load and not self.updated and (not math.isfinite(gap) or gap > OLD_REFERENCE_TOLERANCE):
                raise ConstraintError('Identical policy/reference adapters disagree before update: ' + str(gap))
            from dataclasses import replace
            return replace(action, reference_logprobs=tuple(lp))
        finally: self._switch('policy')

    def update(self, episodes, returns, ledger):
        """One masked GRPO optimizer step; repeated calls allowed within a round."""
        self._ensure(ledger); self._switch('policy')
        old_ref = self._reference_hash()
        def score(sample):
            ledger.before_prefill(len(sample.input_ids), phase='current_training_full_context')
            logits, tensor = self._forward(sample.input_ids, grad=True)
            return masked_causal_logprobs(logits, tensor, sample.allowed)
        optimizer = None
        if self.persistent_optimizer:
            if self._optimizer is None:
                self._optimizer = self.torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.grpo_lr)
            optimizer = self._optimizer
        result = one_update_masked(self.model.parameters(), episodes, returns, score, before_step=ledger.check, lr=self.grpo_lr,
                                   optimizer=optimizer)
        if result['optimizer_updates'] == 1: self.updated = True
        if self._reference_hash() != old_ref: raise RuntimeError('Reference adapter changed during update')
        if state_fingerprint({n: p for n, p in self.model.named_parameters() if '.policy.' not in n}) != self.fixed_hash:
            raise RuntimeError('Frozen base/reference parameters changed')
        result.pop('episodes', None); result.pop('record_hashes', None)
        result['policy_hash_after'] = self._policy_hash()
        self.updates.append(result)
        return result

    def sft_policy(self, rows, ledger, *, epochs, seed, log_path=None):
        """Supervised update of the policy adapter on (messages, action) rows."""
        self._ensure(ledger); self._switch('policy')
        torch = self.torch
        samples = [sft_row(self.tokenizer, row, max_tokens=SFT_CONFIG['max_tokens']) for row in rows]
        params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        if not params or not samples: raise ValueError('Trainable policy and rows required')
        optimizer = torch.optim.AdamW(params, lr=self.sft_lr)
        rng = np.random.default_rng(seed)
        before = self._policy_hash(); stats = []
        self.model.train()
        try:
            for epoch in range(epochs):
                order = rng.permutation(len(samples))
                batches = [order[i:i + SFT_CONFIG['minibatch']] for i in range(0, len(order), SFT_CONFIG['minibatch'])]
                for start in range(0, len(batches), SFT_CONFIG['gradient_accumulation']):
                    group = batches[start:start + SFT_CONFIG['gradient_accumulation']]
                    optimizer.zero_grad(set_to_none=True); losses = []
                    for indices in group:
                        batch = [samples[i] for i in indices]
                        n = sum(len(s['input_ids']) for s in batch)
                        ledger.before_prefill(n, phase='policy_sft')
                        loss = self._sft_loss(batch)
                        (loss / len(group)).backward(); losses.append(float(loss.detach()))
                    norm = torch.nn.utils.clip_grad_norm_(params, SFT_CONFIG['clip_norm'], error_if_nonfinite=True)
                    optimizer.step()
                    stats.append(dict(epoch=epoch, loss=float(np.mean(losses)), gradient_norm=float(norm)))
        finally:
            self.model.eval(); optimizer.zero_grad(set_to_none=True)
        after = self._policy_hash()
        if after == before: raise RuntimeError('Policy SFT did not change parameters')
        self.updated = True
        if state_fingerprint({n: p for n, p in self.model.named_parameters() if '.policy.' not in n}) != self.fixed_hash:
            raise RuntimeError('Frozen base/reference parameters changed during SFT')
        out = dict(rows=len(samples), epochs=epochs, optimizer_updates=len(stats), first_loss=stats[0]['loss'],
                   last_loss=stats[-1]['loss'], policy_hash_before=before, policy_hash_after=after)
        if log_path is not None: save_json(log_path, dict(out, updates=stats))
        return out

    def _sft_loss(self, batch):
        torch = self.torch
        length = max(len(s['input_ids']) for s in batch)
        pad = self.tokenizer.pad_token_id
        ids = torch.tensor([s['input_ids'] + [pad] * (length - len(s['input_ids'])) for s in batch], device=self.run_device)
        labels = torch.tensor([s['labels'] + [-100] * (length - len(s['labels'])) for s in batch], device=self.run_device)
        mask = torch.tensor([[1] * len(s['input_ids']) + [0] * (length - len(s['input_ids'])) for s in batch], device=self.run_device)
        output = self.model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
        if not torch.isfinite(output.loss): raise FloatingPointError('Nonfinite SFT loss')
        return output.loss


def sft_from_base(model_path, qwen_manifest, rows, output, ledger, *, device, epochs, seed, dtype=None):
    """Fresh r16/a32 q,v LoRA on the frozen base; save, unload, exact reload."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict
    verify_qwen(model_path, qwen_manifest)
    set_seed(seed)
    dtype = dtype or ('bfloat16' if device == 'cuda' else 'float32')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)   # MACE (D0 collection) leaves float64 as the global default
    try:
        base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                                    dtype=getattr(torch, dtype), attn_implementation='sdpa')
        model = get_peft_model(base, LoraConfig(**LORA))
    finally:
        torch.set_default_dtype(previous)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if 'lora_' not in name: raise ValueError('Unexpected trainable base parameter')
            parameter.data = parameter.data.float()
        elif parameter.is_floating_point() and parameter.dtype != getattr(torch, dtype):
            raise ValueError('Frozen base must keep its load dtype')
    model.to(device)
    samples = [sft_row(tokenizer, row, max_tokens=SFT_CONFIG['max_tokens']) for row in rows]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=SFT_CONFIG['learning_rate'])
    rng = np.random.default_rng(seed); stats = []
    model.train()
    with Phase(ledger, 'd0_sft_train', rows=len(samples), epochs=epochs):
        for epoch in range(epochs):
            order = rng.permutation(len(samples))
            batches = [order[i:i + SFT_CONFIG['minibatch']] for i in range(0, len(order), SFT_CONFIG['minibatch'])]
            for start in range(0, len(batches), SFT_CONFIG['gradient_accumulation']):
                group = batches[start:start + SFT_CONFIG['gradient_accumulation']]
                optimizer.zero_grad(set_to_none=True); losses = []
                for indices in group:
                    batch = [samples[i] for i in indices]
                    ledger.before_prefill(sum(len(s['input_ids']) for s in batch), phase='d0_sft')
                    length = max(len(s['input_ids']) for s in batch)
                    pad = tokenizer.pad_token_id
                    ids = torch.tensor([s['input_ids'] + [pad] * (length - len(s['input_ids'])) for s in batch], device=device)
                    labels = torch.tensor([s['labels'] + [-100] * (length - len(s['labels'])) for s in batch], device=device)
                    mask = torch.tensor([[1] * len(s['input_ids']) + [0] * (length - len(s['input_ids'])) for s in batch], device=device)
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite SFT loss')
                    (loss / len(group)).backward(); losses.append(float(loss.detach()))
                norm = torch.nn.utils.clip_grad_norm_(params, SFT_CONFIG['clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                stats.append(dict(epoch=epoch, loss=float(np.mean(losses)), gradient_norm=float(norm)))
    model.eval()
    saved = Path(output) / 'adapter'
    expected = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    model.save_pretrained(saved, safe_serialization=True)
    base = model.unload()
    reloaded = PeftModel.from_pretrained(base, saved, is_trainable=False, local_files_only=True)
    actual = get_peft_model_state_dict(reloaded)
    exact = set(actual) == set(expected) and all(torch.equal(actual[k].detach().cpu(), expected[k]) for k in expected)
    if not exact: raise RuntimeError('D0 SFT adapter failed exact reload')
    return dict(rows=len(samples), epochs=epochs, optimizer_updates=len(stats), first_loss=stats[0]['loss'],
                last_loss=stats[-1]['loss'], updates=stats, adapter_hashes=adapter_hashes(saved),
                save_unload_reload=True)
