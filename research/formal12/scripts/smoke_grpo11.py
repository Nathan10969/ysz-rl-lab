"""Review11 constrained-decoding GRPO smoke; import starts no work.

Reuses the frozen v2.2 runner (smoke_grpo10.run_smoke), environment and batch
MACE physics byte-for-byte. Only the actor changes: generation, reference
scoring and the policy update all use the legal-action token trie
(constrained.py). The user approved on 2026-10-25: constrained decoding for all
LLM arms, reuse of the SFT661 adapter although it failed the 76/80 legality
gate (24/80), and one GRPO smoke capped at 2 card-hours.
"""
from __future__ import annotations

import time
PROCESS_STARTED_AT = time.monotonic()
import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

import claude08_env as env
from claude08_data import sha256
from claude08_grpo import state_fingerprint
import smoke_grpo08 as old
import smoke_grpo10 as grpo10
from smoke_grpo08 import (SEED, MAX_NEW_TOKENS, MAX_PROMPT_TOKENS, MAX_TOTAL_TOKENS, QWEN_WEIGHT_SHA,
                          INTERNAL_SECONDS, GeneratedAction, HFActor, decode_action_tokens,
                          require_single_gpu_job, save_json, track_phase)
from constrained import (POLICY_PROTOCOL, OLD_REFERENCE_TOLERANCE, SAMPLING_RECOMPUTE_STOP,
                         ActionTrie, ConstrainedActionSample, ConstraintError,
                         masked_causal_logprobs, one_update_masked)

# Token caps only: v2.2 caps (prefill 293632, seen 262144) do not fit the
# 128-action G4/K8 smoke with full-context old/reference/current scoring of
# ~700-token prompts. Card-hour cost is bounded by Slurm, not by these.
LIMITS = env.BudgetLimits(max_attempts=8, max_relax_actual=132, max_raw_ef=50000,
        max_generated_tokens=9472, max_prefill_tokens=700000, max_seen_tokens=700000,
        wall_seconds=INTERNAL_SECONDS)
ADAPTER661 = {
    'adapter_model.safetensors': 'b17f8fdadbe58d2937f3e55fdc1bf0b5f80a3e1b4c9c47858948e3816b9cddf1',
    'adapter_config.json': '5528babc55c33d4e2edf331cfe49df260173dab169c5851909dd0056f6f54685'}
AUTHORIZATION = dict(review=11, user_approved='2026-10-25', method=POLICY_PROTOCOL,
                     adapter_job='661', adapter_original_gate='FAILED_76of80_gate_observed_24of80',
                     max_card_hours=2, pool='original_10_card_hour_smoke_pool', retries=0)
SOURCES = ('smoke_grpo11.py', 'constrained.py', 'smoke_grpo10.py', 'smoke_grpo08.py', 'claude08_grpo.py',
           'claude08_env.py', 'claude08_data.py', 'claude10_physics.py', 'claude10_data.py',
           'smoke_teacher10.py', 'smoke_sft10.py', 'smoke_mace.py', 'smoke_common.py',
           'calibration.py', 'lattice.py')


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name: sha256(root / name) for name in SOURCES}


@dataclass(frozen=True)
class ConstrainedGeneratedAction(GeneratedAction):
    allowed: tuple[tuple[int, ...], ...] = ()

    def sample(self):
        return ConstrainedActionSample(self.input_ids, self.action_mask, self.old_logprobs,
                                       self.reference_logprobs, self.allowed)


class ConstrainedHFActor(HFActor):
    """HFActor with one prefix mask shared by sampling, old, reference, current.

    ``device='cpu'`` exists only for the explicit local CPU model preflight;
    the production CLI always uses one Slurm CUDA device in BF16. CPU defaults
    to FP32 because BF16 backward on CPU takes minutes per sample.
    """

    def __init__(self, model_path, adapter_path, qwen_manifest, *, release, output, device='cuda', dtype=None):
        super().__init__(model_path, adapter_path, qwen_manifest, release=release, output=output)
        self.run_device = device
        self.dtype_name = dtype or ('bfloat16' if device == 'cuda' else 'float32')
        self.records = []
        self.identical_adapters_at_load = None
        self.policy_load_max_abs_difference = None

    def _ensure(self, ledger):
        if self.model is None:
            with track_phase(ledger, 'HF_load'): self._load(ledger)
        else: self.to_device(self.run_device, ledger)

    def _load(self, ledger):
        ledger.check()
        if self.run_device == 'cuda': require_single_gpu_job()
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        provenance = json.loads(self.qwen_manifest.read_text())
        if provenance.get('repository') != 'Qwen/Qwen3-0.6B' or provenance.get('status') != 'verified_complete':
            raise ValueError('Only verified official Qwen3-0.6B is approved')
        if not any(f['path'] == 'model.safetensors' and f['sha256'] == QWEN_WEIGHT_SHA for f in provenance['files']):
            raise ValueError('Qwen base weight SHA differs from approved manifest')
        for file in provenance['files']:
            ledger.check()
            if sha256(self.model_path / file['path']) != file['sha256']:
                raise ValueError('Qwen file hash changed: ' + file['path'])
        configuration = json.loads((self.adapter_path / 'adapter_config.json').read_text())
        if (configuration.get('r') != 16 or configuration.get('lora_alpha') != 32
                or set(configuration.get('target_modules', [])) != {'q_proj', 'v_proj'}):
            raise ValueError('661 adapter is not the fixed r16 alpha32 q/v LoRA')
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=False)
        base = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=False,
                                                    dtype=getattr(torch, self.dtype_name), attn_implementation='sdpa')
        ledger.check()
        self.model = PeftModel.from_pretrained(base, self.adapter_path, adapter_name='reference',
                                               is_trainable=False, local_files_only=True)
        self.model.load_adapter(self.adapter_path, adapter_name='policy', is_trainable=True, local_files_only=True)
        self._switch('policy')
        # PEFT 0.17.1 load_adapter rounds the second adapter through BF16
        # (~1e-4), while the first keeps the saved FP32 values. Verify the
        # reference against the file, then give the policy those exact values.
        from safetensors.torch import load_file
        saved = load_file(str(self.adapter_path / 'adapter_model.safetensors'))
        named = dict(self.model.named_parameters())
        references = {n: n.replace('.reference.', '.') for n in named if '.reference.' in n}
        if (len(references) != len(saved) or set(references.values()) != set(saved)
                or not all(torch.equal(named[n], saved[k].to(named[n].dtype)) for n, k in references.items())):
            raise RuntimeError('Reference adapter differs from the saved 661 tensors')
        pairs = [(n, n.replace('.policy.', '.reference.')) for n in named if '.policy.' in n]
        if len(pairs) != len(references) or any(other not in named for _, other in pairs):
            raise RuntimeError('Policy/reference adapter parameter sets differ')
        self.policy_load_max_abs_difference = max(float((named[n].float() - named[o].float()).abs().max()) for n, o in pairs)
        with torch.no_grad():
            for n, other in pairs: named[n].copy_(named[other])
        self.identical_adapters_at_load = all(torch.equal(named[n], named[o]) for n, o in pairs)
        if not self.identical_adapters_at_load:
            raise RuntimeError('Policy and reference adapters differ at load')
        self.initial_reference_hash = self._reference_hash()
        self.initial_policy_hash = self._policy_hash()
        self.fixed_hash = state_fingerprint({n: p for n, p in self.model.named_parameters() if '.policy.' not in n})
        self.to_device(self.run_device, ledger)
        self.generator = torch.Generator(device=self.run_device).manual_seed(SEED)

    def _forward(self, ids, *, grad=False):
        torch = self.torch
        tensor = torch.tensor([list(ids)], device=self.run_device)
        if grad:
            output = self.model(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False)
        else:
            with torch.no_grad():
                output = self.model(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False)
        return output.logits, tensor

    def generate(self, messages, ledger, *, phase, occupancy):
        self._ensure(ledger); self._switch('policy')
        if self._policy_hash() != self.initial_policy_hash and not self.updated:
            raise RuntimeError('Old generation policy changed before the single update')
        trie = ActionTrie(self.tokenizer, occupancy, max_new_tokens=MAX_NEW_TOKENS)
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False)
        if hasattr(ids, 'tolist'): ids = ids.tolist()
        if not ids or len(ids) > MAX_PROMPT_TOKENS: raise ValueError('Prompt exceeds 1984; no silent truncation')
        torch = self.torch
        inputs = torch.tensor([ids], device=self.run_device)
        ledger.before_prefill(len(ids), phase=phase + '_generation_prefill')
        generated, sampled_lp, allowed_rows = [], [], []
        allowed_mass, argmax_allowed = [], []
        with torch.no_grad():
            output = self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=True)
            cache = output.past_key_values
            for index in range(MAX_NEW_TOKENS):
                allowed = trie.allowed(generated)
                scores = output.logits[0, -1].float()
                if not torch.isfinite(scores).all(): raise FloatingPointError('Nonfinite generation logits')
                index_tensor = torch.tensor(allowed, device=self.run_device)
                subset = scores.index_select(0, index_tensor)
                masked = subset.log_softmax(-1)
                allowed_mass.append(float(scores.log_softmax(-1).index_select(0, index_tensor).exp().sum()))
                argmax_allowed.append(int(scores.argmax()) in allowed)
                ledger.before_generated_token(phase=phase + '_sample')
                # Forced positions draw no random number.
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
        action = trie.action(generated)
        text = decode_action_tokens(self.tokenizer, generated)
        if text != action.to_json(): raise ConstraintError('Decoded text differs from canonical action text')
        try: env.apply_action(occupancy, env.parse_action(text))
        except env.ActionError as exc: raise ConstraintError('Constrained decoder produced illegal action') from exc
        full = tuple(ids + generated)
        if len(full) > MAX_TOTAL_TOKENS: raise ValueError('Total sequence exceeds 2048')
        aligned = tuple([()] * (len(ids) - 1) + allowed_rows)
        # Old-policy log-probs come from the same full-context forward used by
        # reference and update scoring, not from the KV-cached sampling logits.
        ledger.before_prefill(len(full), phase=phase + '_policy_full_context')
        logits, tensor = self._forward(full)
        lp = masked_causal_logprobs(logits, tensor, aligned).cpu().tolist()
        del logits
        action_lp = lp[len(ids) - 1:]
        gap = max(abs(a - b) for a, b in zip(action_lp, sampled_lp))
        self.records.append(dict(phase=phase, text=text, prompt_tokens=len(ids), action_tokens=len(generated),
            branch_tokens=sum(len(row) > 1 for row in allowed_rows), allowed_sizes=[len(r) for r in allowed_rows],
            allowed_probability_mass=allowed_mass, unconstrained_argmax_allowed=argmax_allowed,
            legal_set_size=len(trie), trie_sha256=trie.digest, sampling_vs_recompute_max_abs=gap))
        if not math.isfinite(gap) or gap > SAMPLING_RECOMPUTE_STOP:
            raise ConstraintError('KV-cached sampling and full-context old log-probs disagree: ' + str(gap))
        mask = tuple([False] * len(ids) + [True] * len(generated))
        return ConstrainedGeneratedAction(text, full, mask, tuple(lp), tuple(lp), aligned)

    def score_reference(self, action, ledger):
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
            if not self.updated and (not math.isfinite(gap) or gap > OLD_REFERENCE_TOLERANCE):
                raise ConstraintError('Identical policy/reference adapters disagree before update: ' + str(gap))
            return replace(action, reference_logprobs=tuple(lp))
        finally: self._switch('policy')

    def update(self, episodes, returns, ledger):
        self._ensure(ledger); self._switch('policy')
        old_ref = self._reference_hash(); old_policy = self._policy_hash()
        if old_policy != self.initial_policy_hash: raise RuntimeError('Old policy changed before update')
        def score(sample):
            ledger.before_prefill(len(sample.input_ids), phase='current_training_full_context')
            logits, tensor = self._forward(sample.input_ids, grad=True)
            return masked_causal_logprobs(logits, tensor, sample.allowed)
        result = one_update_masked(self.model.parameters(), episodes, returns, score, before_step=ledger.check)
        self.updated = result['optimizer_updates'] == 1
        if self._reference_hash() != old_ref: raise RuntimeError('Reference adapter changed during update')
        if state_fingerprint({n: p for n, p in self.model.named_parameters() if '.policy.' not in n}) != self.fixed_hash:
            raise RuntimeError('Frozen base/reference parameters changed')
        result.update(reference_hash_before=old_ref, reference_hash_after=self._reference_hash(),
                      old_policy_hash=old_policy, current_policy_hash=self._policy_hash())
        return result


def summarize(records):
    """Intervention diagnostics with explicit denominators."""
    tokens = [(m, a, s) for r in records for m, a, s in
              zip(r['allowed_probability_mass'], r['unconstrained_argmax_allowed'], r['allowed_sizes'])]
    branch = [(m, a) for m, a, s in tokens if s > 1]
    def block(rows):
        if not rows: return dict(denominator=0)
        return dict(denominator=len(rows),
                    unconstrained_argmax_outside_allowed_fraction=sum(not a for _, a in rows) / len(rows),
                    masked_out_probability_mass_mean=sum(1 - m for m, _ in rows) / len(rows),
                    masked_out_probability_mass_max=max(1 - m for m, _ in rows))
    gaps = [r['sampling_vs_recompute_max_abs'] for r in records]
    ref = [r['old_vs_reference_max_abs'] for r in records if 'old_vs_reference_max_abs' in r]
    return dict(policy_protocol=POLICY_PROTOCOL, actions=len(records),
                phases={p: sum(r['phase'] == p for r in records) for p in sorted({r['phase'] for r in records})},
                all_generated_tokens=block([(m, a) for m, a, _ in tokens]), branching_tokens=block(branch),
                interpretation='diagnostic on the constrained prefix path; not an unconstrained legality rate',
                sampling_vs_recompute_max_abs=max(gaps) if gaps else None,
                old_vs_reference_max_abs=max(ref) if ref else None, old_vs_reference_count=len(ref),
                old_reference_tolerance=OLD_REFERENCE_TOLERANCE, sampling_recompute_stop=SAMPLING_RECOMPUTE_STOP,
                baseline_unconstrained_SFT661_legality='24/80')


def gate(result, records):
    """Review11 smoke gate (replaces the v2.2 19/20 legality gate)."""
    legality = result.get('legality', {})
    update = result.get('update', {})
    checks = dict(
        legality_20_of_20=legality.get('legal') == 20 and len(legality.get('generations', [])) == 20,
        completed_one_update=result.get('status') == 'COMPLETED_ONE_UPDATE' and result.get('optimizer_updates') == 1,
        finite_gradients=update.get('finite_gradients') is True,
        parameter_changed=isinstance(update.get('parameter_delta_l2'), (int, float)) and update['parameter_delta_l2'] > 0,
        save_unload_reload=result.get('adapter_check', {}).get('save_unload_reload') is True,
        old_reference_within_tolerance=bool(records) and all(
            r.get('old_vs_reference_max_abs', 0.) <= OLD_REFERENCE_TOLERANCE for r in records)
            and sum('old_vs_reference_max_abs' in r for r in records) == 128,
        usable_for_workflow=result.get('usable_for_workflow') is True)
    return dict(passed=all(checks.values()), checks=checks)


def verify_release(path):
    """Hash-bound Review11 release; reads files only, grants no submission."""
    path = Path(path).resolve(); root = path.parent
    manifest = json.loads(path.read_text())
    def checked(reference):
        if (not isinstance(reference, dict) or Path(reference.get('path', '/')).is_absolute()
                or len(reference.get('sha256', '')) != 64):
            raise ValueError('Release references require relative path and SHA256')
        target = root / reference['path']
        if sha256(target) != reference['sha256']: raise ValueError('Release hash mismatch: ' + reference['path'])
        return target
    if manifest.get('schema') != 'review11-grpo-release-v1' or manifest.get('seed') != SEED:
        raise ValueError('Review11 release schema/seed required')
    if manifest.get('authorization') != AUTHORIZATION: raise ValueError('Review11 authorization changed')
    if manifest.get('source_hashes') != source_hashes(): raise ValueError('Review11 source hashes changed')
    frozen = grpo10.source_hashes()
    if (frozen['claude08_env.py'] != grpo10.REVIEWED_ENV_SHA or frozen['claude10_physics.py'] != grpo10.REVIEWED_PHYSICS_SHA
            or any(frozen[k] != v for k, v in grpo10.REVIEWED_PIPELINE_SHAS.items())):
        raise ValueError('Frozen v2.2 environment/physics/pipeline sources changed')
    protocol_sha = env.protocol_fingerprint(env.RelaxationSpec())
    if manifest.get('protocol') != env.PROTOCOL or manifest.get('protocol_sha') != protocol_sha:
        raise ValueError('v2.2 physics protocol changed')
    adapter = {}
    for reference in manifest['adapter_files']:
        adapter[Path(checked(reference)).name] = reference['sha256']
    if adapter != ADAPTER661: raise ValueError('Adapter is not the hash-bound SFT661 adapter')
    data = checked(manifest['data_manifest']); qwen = checked(manifest['qwen_manifest'])
    checkpoint = checked(manifest['checkpoint'])
    if manifest['checkpoint']['sha256'] != env.MODEL_SHA: raise ValueError('MACE checkpoint changed')
    provenance = json.loads(qwen.read_text())
    if (provenance.get('status') != 'verified_complete'
            or not any(f.get('path') == 'model.safetensors' and f.get('sha256') == QWEN_WEIGHT_SHA
                       for f in provenance.get('files', []))):
        raise ValueError('Verified Qwen manifest required')
    preflight = json.loads(checked(manifest['cpu_model_preflight']).read_text())
    if (preflight.get('status') != 'passed' or preflight.get('source_hashes') != source_hashes()
            or preflight.get('adapter_hashes') != ADAPTER661):
        raise ValueError('Passed Review11 CPU model preflight on these sources required')
    return dict(verified=True, manifest_sha256=sha256(path), adapter_hashes=adapter,
                protocol=env.PROTOCOL, protocol_sha=protocol_sha, policy_protocol=POLICY_PROTOCOL,
                source_hashes=source_hashes(), data_manifest_sha256=manifest['data_manifest']['sha256'],
                qwen_manifest_sha256=manifest['qwen_manifest']['sha256'],
                checkpoint_sha256=manifest['checkpoint']['sha256'], authorization=AUTHORIZATION,
                data_path=str(data), checkpoint_path=str(checkpoint))


def load_grpo_data(path):
    """Hash-bound frozen v2.2 data; full semantic recheck of GRPO rows only.

    claude10_data.load_frozen_data also re-verifies the 97 MB teacher result
    and all 4096 SFT format rows (minutes of idle GPU allocation). Those files
    were independently verified at freeze; here they are bound by hash, and
    only the 4 starts and 20 legality states the runner consumes are rebuilt.
    """
    from claude10_data import DATA_SCHEMA, PROTOCOL_SHA, PURPOSE, source_hashes as data_sources
    from claude08_data import unscored_state
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if (manifest.get('schema') != DATA_SCHEMA or manifest.get('seed') != SEED
            or manifest.get('protocol') != env.PROTOCOL or manifest.get('protocol_sha') != PROTOCOL_SHA
            or manifest.get('checkpoint_sha256') != env.MODEL_SHA or manifest.get('purpose') != PURPOSE
            or manifest.get('source_hashes') != data_sources()):
        raise ValueError('Frozen v2.2 data/source binding changed')
    for name, digest in manifest['files'].items():
        if sha256(path / name) != digest: raise ValueError('Frozen data hash changed: ' + name)
    rows = {}
    for name, count in (('starts.jsonl', 4), ('legality_grpo.jsonl', 20)):
        rows[name] = [json.loads(line) for line in (path / name).read_text().splitlines()]
        if len(rows[name]) != count: raise ValueError('Frozen count changed: ' + name)
        for row in rows[name]:
            occupancy = env.validate_occupancy(row['occupancy'])
            if (row['canonical_key'] != env.occupancy_key(occupancy) or row.get('unscored') is not True
                    or row.get('energy_eV') is not None or row.get('physical_calls') != 0
                    or row['messages'] != env.encode_prompt_v2(unscored_state(occupancy))):
                raise ValueError('Unscored GRPO row changed')
    return {'manifest': manifest, 'rows': rows, 'manifest_sha256': sha256(path / 'manifest.json')}


def run(actor, physics, starts, heldout, *, output, started_at=None, runtime_context=None):
    """Frozen v2.2 run_smoke with Review11 caps, then diagnostics and gate."""
    grpo10.LIMITS = LIMITS  # _validate_protocol bounds injected limits by this
    result = grpo10.run_smoke(actor, physics, starts, heldout, output=output, limits=LIMITS,
                              started_at=started_at, runtime_context=runtime_context)
    summary = summarize(actor.records)
    verdict = gate(result, actor.records)
    if output is not None:
        save_json(Path(output) / 'constrained_records.json', actor.records)
        save_json(Path(output) / 'review11_gate.json', dict(gate=verdict, diagnostics=summary,
            status=result.get('status'), counters=result.get('counters'), wall_seconds=result.get('wall_seconds'),
            identical_adapters_at_load=actor.identical_adapters_at_load,
            policy_load_max_abs_difference=actor.policy_load_max_abs_difference, source_hashes=source_hashes()))
    return result, summary, verdict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'adapter', 'qwen-manifest', 'checkpoint', 'release-manifest', 'data', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists(): raise FileExistsError(args.output)
    release = verify_release(args.release_manifest)
    for target, key in ((args.data / 'manifest.json', 'data_manifest_sha256'),
                        (args.qwen_manifest, 'qwen_manifest_sha256'), (args.checkpoint, 'checkpoint_sha256')):
        if sha256(target) != release[key]: raise ValueError('Runtime input differs from released ' + key)
    old.verify_adapter_binding(args.adapter, release)
    frozen = load_grpo_data(args.data)
    runtime = require_single_gpu_job()
    if 'A100' not in runtime.get('device_name', ''): raise RuntimeError('Review11 requires one Slurm CUDA A100')
    actor = ConstrainedHFActor(args.model, args.adapter, args.qwen_manifest, release=release, output=args.output)
    from claude10_physics import BatchMACEPhysics
    class LazyPhysics:
        def __init__(self): self.engine = None
        def relax_many(self, requests, counter, max_steps=300):
            if self.engine is None:
                self.engine = BatchMACEPhysics(args.checkpoint, args.output / 'physics_engine', device='cuda', actor=actor)
            return self.engine.relax_many(requests, counter, max_steps=max_steps)
        @property
        def last_evidence(self): return [] if self.engine is None else self.engine.last_evidence
        def close(self):
            if self.engine is not None: self.engine.close()
    result, summary, verdict = run(actor, LazyPhysics(), [r['occupancy'] for r in frozen['rows']['starts.jsonl']],
        frozen['rows']['legality_grpo.jsonl'], output=args.output, started_at=PROCESS_STARTED_AT,
        runtime_context=dict(runtime=runtime, release=release, data_manifest_sha256=frozen['manifest_sha256']))
    print(json.dumps(dict(status=result['status'], gate=verdict, counters=result['counters'],
                          wall_seconds=result['wall_seconds'])))
    if not verdict['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
