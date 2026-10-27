"""Fixed Review10 v2 SFT. No imports, downloads, model loads or writes on import."""
from __future__ import annotations
import time
PROCESS_STARTED_AT = time.monotonic()
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import numpy as np

import claude08_env as env
from claude08_data import action_sft_row, sha256
from claude10_data import SEED, PURPOSE, load_frozen_data, source_hashes, write_json, persist_result, PROTOCOL_SHA, SFT_SCHEMA
from smoke_grpo08 import QWEN_WEIGHT_SHA, decode_action_tokens, require_single_gpu_job

CONFIG = {'epochs':2, 'minibatch':4, 'gradient_accumulation':2, 'learning_rate':1e-4,
          'clip_norm':1., 'lora_r':16, 'lora_alpha':32, 'lora_dropout':.05,
          'target_modules':['q_proj','v_proj'], 'teacher_oversample':8,
          'max_prompt_tokens':1984, 'max_sequence_tokens':2048, 'max_new_tokens':64,
          'validation_denominator':80, 'required_legal':76, 'wall_seconds':6000,
          'outer_seconds':6600,'slurm_seconds':6660,
          'base_dtype':'bfloat16', 'adapter_dtype':'float32',
          'temperature':1., 'top_p':1., 'top_k':0, 'constrained_decoding':False}


class RuntimeGuard:
    def __init__(self, *, started_at=None, clock=time.monotonic):
        self.clock = clock
        self.started_at = clock() if started_at is None else started_at
    def check(self):
        elapsed = self.clock()-self.started_at
        if not np.isfinite(elapsed) or elapsed < 0 or elapsed >= CONFIG['wall_seconds']:
            raise env.BudgetExceeded('Review10 v2.2 SFT entire-stage 6000-second cap')


def build_schedule(samples):
    """Freeze two deterministic epochs including partial last accumulation groups."""
    rng = np.random.default_rng(SEED)
    epochs = []
    padded, nonpadding, action = 0, 0, 0
    for epoch in range(CONFIG['epochs']):
        order = list(map(int, rng.permutation(len(samples))))
        batches = [order[i:i+CONFIG['minibatch']] for i in range(0,len(order),CONFIG['minibatch'])]
        for batch in batches:
            padded += len(batch) * max(len(samples[i]['input_ids']) for i in batch)
            nonpadding += sum(len(samples[i]['input_ids']) for i in batch)
            action += sum(samples[i]['action_tokens'] for i in batch)
        epochs.append(batches)
    return {'epochs':epochs, 'microbatches':sum(map(len,epochs)),
            'optimizer_updates':sum(math.ceil(len(b)/CONFIG['gradient_accumulation']) for b in epochs),
            'seen_padded_tokens':padded, 'seen_nonpadding_tokens':nonpadding,
            'action_tokens':action, 'scheduled_examples':len(samples)*CONFIG['epochs']}


def tokenizer_preflight(tokenizer, frozen):
    rows = frozen['rows']
    if len(rows['format_train.jsonl']) != 4096 or len(rows['legality_sft.jsonl']) != 80:
        raise ValueError('Fixed 4096/80 data counts required')
    training = rows['format_train.jsonl'] + rows['teacher_train.jsonl'] * CONFIG['teacher_oversample']
    samples = []
    for row in training:
        sample = action_sft_row(tokenizer, row, max_tokens=2048)
        if sample['prompt_tokens'] > 1984: raise ValueError('SFT prompt exceeds 1984; no truncation')
        samples.append(sample)
    validation_prompts = []
    for row in rows['legality_sft.jsonl']:
        ids = tokenizer.apply_chat_template(row['messages'], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
        if hasattr(ids,'tolist'): ids=ids.tolist()
        if not ids or len(ids) > 1984: raise ValueError('Validation prompt exceeds 1984; no truncation')
        validation_prompts.append(len(ids))
    schedule = build_schedule(samples)
    plan = {'config':CONFIG,'protocol':env.PROTOCOL,'protocol_sha':PROTOCOL_SHA, 'data_manifest_sha256':frozen['manifest_sha256'],
            'source_hashes':source_hashes(), 'schedule':schedule,
            'unique_format_examples':4096, 'independent_accepted_teacher_train_examples':len(rows['teacher_train.jsonl']),
            'teacher_repeated_examples':len(rows['teacher_train.jsonl'])*8,
            'teacher_repeat_interpretation':'eight_repeats_not_independent_data',
            'max_sequence_tokens':max(len(s['input_ids']) for s in samples),
            'max_prompt_tokens':max(s['prompt_tokens'] for s in samples),
            'validation_prompt_tokens':validation_prompts,
            'max_validation_generated_tokens':80*64,
            'seen_token_ceiling':schedule['seen_padded_tokens']+sum(validation_prompts)+80*64,
            'loss':'action_and_EOS_only_prompt_and_padding_minus100',
            'base':'frozen_official_Qwen3-0.6B_without_633_adapter', 'physical_calls':0}
    return samples, plan


def collate_rows(samples, *, pad_token_id, device):
    import torch
    if not samples or type(pad_token_id) is not int: raise ValueError('Samples and integer pad token required')
    length = max(len(s['input_ids']) for s in samples)
    inputs, labels, attention = [], [], []
    for sample in samples:
        padding = length-len(sample['input_ids'])
        inputs.append(sample['input_ids']+[pad_token_id]*padding)
        labels.append(sample['labels']+[-100]*padding)
        attention.append([1]*len(sample['input_ids'])+[0]*padding)
    return {'input_ids':torch.tensor(inputs,device=device,dtype=torch.long),
            'labels':torch.tensor(labels,device=device,dtype=torch.long),
            'attention_mask':torch.tensor(attention,device=device,dtype=torch.long)}


def _fingerprint(parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(-1).view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def train_updates(model, optimizer, samples, plan, guard, *, pad_token_id, device, log_path=None):
    """Every planned update requires finite nonzero gradients and a real tensor change."""
    import torch
    if plan != build_schedule(samples): raise ValueError('Frozen training schedule changed')
    parameters = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not parameters: raise ValueError('No trainable adapter parameters')
    before = _fingerprint(parameters)
    result = {'optimizer_updates':0, 'microbatches':0, 'seen_padded_tokens':0,
              'seen_nonpadding_tokens':0, 'action_tokens':0, 'updates':[]}
    model.train()
    for epoch, batches in enumerate(plan['epochs']):
        for start in range(0,len(batches),CONFIG['gradient_accumulation']):
            group = batches[start:start+CONFIG['gradient_accumulation']]
            optimizer.zero_grad(set_to_none=True)
            update_before = _fingerprint(parameters)
            losses = []
            for indices in group:
                guard.check()
                batch_samples = [samples[i] for i in indices]
                batch = collate_rows(batch_samples, pad_token_id=pad_token_id, device=device)
                n = batch['input_ids'].numel()
                if result['seen_padded_tokens']+n > plan['seen_padded_tokens']:
                    raise env.BudgetExceeded('Frozen SFT token ceiling')
                result['seen_padded_tokens'] += n
                result['seen_nonpadding_tokens'] += int(batch['attention_mask'].sum().item())
                result['action_tokens'] += sum(s['action_tokens'] for s in batch_samples)
                output = model(**batch, use_cache=False)
                if not torch.isfinite(output.loss).all(): raise FloatingPointError('Nonfinite training loss')
                (output.loss/len(group)).backward()
                losses.append(float(output.loss.detach()))
                result['microbatches'] += 1
                del output, batch
            guard.check()
            grads = [p.grad for _,p in parameters if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all().item() for g in grads):
                raise FloatingPointError('Missing or nonfinite adapter gradients')
            norm = torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.,error_if_nonfinite=True)
            if not torch.isfinite(norm).item() or norm.item() <= 0: raise FloatingPointError('Zero/nonfinite gradient norm')
            optimizer.step()
            guard.check()
            if _fingerprint(parameters) == update_before: raise RuntimeError('Optimizer update did not change adapter parameters')
            result['optimizer_updates'] += 1
            record = {'update':result['optimizer_updates'], 'epoch':epoch+1,
                      'microbatches':result['microbatches'], 'loss':float(np.mean(losses)),
                      'gradient_norm':float(norm), 'seen_padded_tokens':result['seen_padded_tokens'],
                      'elapsed_seconds':guard.clock()-guard.started_at}
            result['updates'].append(record)
            if log_path is not None:
                with Path(log_path).open('a') as handle: handle.write(json.dumps(record,allow_nan=False)+'\n')
    for key in ('microbatches','optimizer_updates','seen_padded_tokens','seen_nonpadding_tokens','action_tokens'):
        if result[key] != plan[key]: raise RuntimeError('SFT failed to complete frozen schedule: '+key)
    after = _fingerprint(parameters)
    result.update(parameters_changed=before != after, adapter_hash_before=before, adapter_hash_after=after)
    if before == after: raise RuntimeError('No net adapter parameter change')
    return result


def evaluate_legality(rows, generate, guard, *, progress=None, generation_counts=None, journal=None):
    if len(rows) != 80: raise ValueError('Exactly eighty frozen heldout generations required')
    result = {} if progress is None else progress
    result.update(denominator=80, legal=0, attempts=0, per_operation={}, failure_types={}, records=[], status='running')
    counts = {} if generation_counts is None else generation_counts
    def append(event):
        if journal is not None:
            with Path(journal).open('a') as handle: handle.write(json.dumps(event,allow_nan=False)+'\n')
    for index, row in enumerate(rows):
        guard.check()
        result['attempts'] += 1
        record = {'index':index, 'status':'started', 'text':None, 'tokens_before':dict(counts)}
        result['active_attempt'] = record
        append({'event':'started', **record})
        try:
            text = generate(row['messages'])
            record['text'] = text
        except Exception as exc:
            record.update(status='failed', error=type(exc).__name__+': '+str(exc),
                          tokens_after=dict(counts),
                          token_delta={k:v-record['tokens_before'].get(k,0) for k,v in counts.items()})
            result.update(status='failed',failed_attempt=record)
            result.pop('active_attempt',None)
            append({'event':'failed', **record})
            raise
        record.update(status='completed',tokens_after=dict(counts),
                      token_delta={k:v-record['tokens_before'].get(k,0) for k,v in counts.items()})
        try:
            action = env.parse_action(text)
            env.apply_action(row['occupancy'],action)
            result['legal'] += 1
            result['per_operation'][action.op] = result['per_operation'].get(action.op,0)+1
            record.update(legal=True, op=action.op)
        except (ValueError, TypeError, AttributeError) as exc:
            kind = type(exc).__name__+': '+str(exc)
            result['failure_types'][kind] = result['failure_types'].get(kind,0)+1
            record.update(legal=False, failure=kind)
        result['records'].append(record)
        result.pop('active_attempt',None)
        append({'event':'completed', **record})
    guard.check()
    result['status'] = 'passed' if result['legal'] >= 76 else 'NOT_RELEASED_SFT_LEGALITY_GATE'
    return result


def sample_action(model, tokenizer, messages, guard, generator, counts, *, device='cuda', trace=None):
    """Same full-softmax temperature-one distribution used by the PG actor."""
    import torch
    guard.check()
    ids = tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if hasattr(ids,'tolist'): ids=ids.tolist()
    if not ids or len(ids)>1984: raise ValueError('Prompt exceeds 1984; truncation forbidden')
    inputs = torch.tensor([ids],device=device)
    counts['prefill_tokens'] += len(ids)
    counts['seen_tokens'] += len(ids)
    generated = []
    try:
        with torch.no_grad():
            output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),use_cache=True)
            cache = output.past_key_values
            for index in range(64):
                guard.check()
                scores = output.logits[0,-1].float()
                if not torch.isfinite(scores).all(): raise FloatingPointError('Nonfinite full-softmax logits')
                counts['generated_tokens'] += 1
                counts['seen_tokens'] += 1
                logprobs = scores.log_softmax(-1)
                token = int(torch.multinomial(logprobs.exp(),1,generator=generator).item())
                generated.append(token)
                if token == tokenizer.eos_token_id: break
                if index+1 < 64:
                    output = model(input_ids=torch.tensor([[token]],device=device),
                        attention_mask=torch.ones((1,len(ids)+len(generated)),device=device,dtype=torch.long),
                        past_key_values=cache,use_cache=True)
                    cache = output.past_key_values
        guard.check()
        return decode_action_tokens(tokenizer,generated)
    finally:
        if trace is not None:
            trace['generated_token_ids'] = list(generated)
            try: trace['text'] = decode_action_tokens(tokenizer,generated)
            except Exception as exc: trace['decode_error'] = type(exc).__name__+': '+str(exc)


def verify_qwen(model_path, manifest_path, guard):
    provenance = json.loads(Path(manifest_path).read_text())
    if provenance.get('repository') != 'Qwen/Qwen3-0.6B' or provenance.get('status') != 'verified_complete':
        raise ValueError('Verified official BASE Qwen3-0.6B required')
    if not any(f.get('path')=='model.safetensors' and f.get('sha256')==QWEN_WEIGHT_SHA for f in provenance.get('files', [])):
        raise ValueError('Qwen base weight SHA differs from reviewed base')
    root = Path(model_path).resolve()
    if (root/'adapter_config.json').exists(): raise ValueError('SFT must start from the frozen base, without an adapter')
    for file in provenance['files']:
        guard.check()
        path = (root/file['path']).resolve()
        if not path.is_relative_to(root) or sha256(path) != file['sha256']:
            raise ValueError('Official Qwen file changed: '+file['path'])
    guard.check()
    return provenance


def save_reload_adapter(model, output, guard):
    """Save adapter, unload without merging, reload disk tensors and compare exactly."""
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    guard.check()
    saved = Path(output)/'adapter'
    expected = {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    model.save_pretrained(saved,safe_serialization=True)
    guard.check()
    base = model.unload()
    if any('lora_' in name for name,_ in base.named_parameters()): raise RuntimeError('Unloaded base retains LoRA')
    reloaded = PeftModel.from_pretrained(base,saved,is_trainable=False,local_files_only=True)
    actual = get_peft_model_state_dict(reloaded)
    if set(actual) != set(expected) or not all(torch.equal(actual[k].detach().cpu(),expected[k]) for k in expected):
        raise RuntimeError('Saved adapter failed exact disk tensor reload')
    guard.check()
    return reloaded, {'save_unload_reload':True, 'unloaded_without_merge':True,
        'state_exact_match':True, 'reloaded_from_disk':True, 'tensor_count':len(actual),
        'adapter_sha256':sha256(saved/'adapter_model.safetensors'),
        'adapter_config_sha256':sha256(saved/'adapter_config.json')}


def run_sft(model_path, qwen_manifest, data_path, output, *, preflight_only=False, started_at=None):
    guard = RuntimeGuard(started_at=started_at)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    result = {'schema':SFT_SCHEMA,'protocol':env.PROTOCOL,'protocol_sha':PROTOCOL_SHA, 'status':'running', 'purpose':PURPOSE,
              'base_weight_sha256':QWEN_WEIGHT_SHA,'teacher_checkpoint_sha256':env.MODEL_SHA,
              'config':CONFIG, 'seed':SEED, 'physical_calls':0, 'GPU_work_performed':False,
              'source_hashes':source_hashes(), 'validation_retries':0, 'training_runs':0}
    model = None
    base = None
    try:
        guard.check()
        provenance = verify_qwen(model_path,qwen_manifest,guard)
        frozen = load_frozen_data(data_path)
        guard.check()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        samples, plan = tokenizer_preflight(tokenizer,frozen)
        guard.check()
        result.update(data_manifest_sha256=frozen['manifest_sha256'], qwen_manifest_sha256=sha256(qwen_manifest),
                      qwen_repository=provenance['repository'])
        write_json(output/'frozen_schedule.json',plan)
        result['frozen_schedule_sha256'] = sha256(output/'frozen_schedule.json')
        if preflight_only:
            result.update(status='tokenizer_preflight_passed', planned_microbatches=plan['schedule']['microbatches'],
                          planned_optimizer_updates=plan['schedule']['optimizer_updates'],
                          seen_token_ceiling=plan['seen_token_ceiling'])
            return result
        result['runtime'] = require_single_gpu_job()
        guard.check()
        import torch
        from transformers import AutoModelForCausalLM, set_seed
        from peft import LoraConfig, get_peft_model
        set_seed(SEED)
        result['GPU_work_performed'] = True
        base = AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
                    torch_dtype=torch.bfloat16,attn_implementation='sdpa')
        guard.check()
        model = get_peft_model(base, LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
                    target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
        for name,parameter in model.named_parameters():
            if parameter.requires_grad:
                if 'lora_' not in name: raise ValueError('Unexpected trainable base parameter')
                parameter.data = parameter.data.float()
            elif parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
                raise ValueError('Frozen base must remain bfloat16')
        model.to('cuda')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        guard.check()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        result['training_runs'] = 1
        result['training'] = train_updates(model,optimizer,samples,plan['schedule'],guard,
                    pad_token_id=tokenizer.pad_token_id,device='cuda',log_path=output/'updates.jsonl')
        del optimizer
        model, result['reload'] = save_reload_adapter(model,output,guard)
        model.eval()
        generator = torch.Generator(device='cuda').manual_seed(SEED)
        counts = {'prefill_tokens':0,'generated_tokens':0,'seen_tokens':0}
        result['generation_tokens'] = counts
        result['legality'] = {}
        def generate(messages):
            return sample_action(model,tokenizer,messages,guard,generator,counts,
                                 trace=result['legality'].get('active_attempt'))
        evaluate_legality(frozen['rows']['legality_sft.jsonl'],generate,guard,progress=result['legality'],
                          generation_counts=counts,journal=output/'legality_attempts.jsonl')
        result['actual_seen_tokens'] = result['training']['seen_padded_tokens']+counts['seen_tokens']
        if result['actual_seen_tokens'] > plan['seen_token_ceiling']: raise env.BudgetExceeded('Frozen complete-stage token ceiling')
        guard.check()
        result.update(status='completed' if result['legality']['status']=='passed' else 'NOT_RELEASED_SFT_LEGALITY_GATE',
            resources={'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                       'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()},
            adapter_release_allowed=result['legality']['status']=='passed')
        write_json(output/'legality.json',result['legality'])
        return result
    except Exception as exc:
        result.update(status='failed', error=type(exc).__name__+': '+str(exc),adapter_release_allowed=False)
        raise
    finally:
        if 'generation_tokens' in result:
            result['actual_seen_tokens'] = result['training']['seen_padded_tokens']+result['generation_tokens']['seen_tokens']
            write_json(output/'legality.json',result['legality'])
        cleanup_begin = guard.clock()
        cleanup_error = None
        try:
            model = None
            base = None
            gc.collect()
            if result['GPU_work_performed']:
                import torch
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            result['cleanup'] = {'status':'completed','seconds':guard.clock()-cleanup_begin}
        except Exception as exc:
            cleanup_error = exc
            result.update(status='failed',error='SFT cleanup failed: '+str(exc),adapter_release_allowed=False,
                          cleanup={'status':'failed','seconds':guard.clock()-cleanup_begin})
        overrun = persist_result(output/'result.json',result,started_at=guard.started_at,clock=guard.clock,
                                 wall_seconds=CONFIG['wall_seconds'])
        if cleanup_error is not None: raise RuntimeError('SFT cleanup failed') from cleanup_error
        if overrun: raise env.BudgetExceeded('SFT entire-stage runtime cap including cleanup/persist')


def main():
    parser = argparse.ArgumentParser()
    for name in ('model','qwen-manifest','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args = parser.parse_args()
    result = run_sft(args.model,args.qwen_manifest,args.data,args.output,
                     preflight_only=args.preflight_only,started_at=PROCESS_STARTED_AT)
    print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds']}))
    if result['status'] not in ('completed','tokenizer_preflight_passed'): raise SystemExit(1)


if __name__ == '__main__': main()
