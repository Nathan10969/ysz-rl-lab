"""One tiny bounded LoRA SFT run; no RSI, scoring, or model-size comparison."""
import argparse
import json
import time
import traceback
from pathlib import Path
from smoke_common import SMOKE, Budget, distribution, require_cuda_job, resources, save_json, sha256, versions

LIMITS = {"optimization_steps": 32, "gradient_accumulation": 2, "microbatches": 64,
          "max_sequence_tokens": 512, "seen_token_cap": 32768, "train_seconds": 900,
          "validation_prompts": 4, "generation_tokens_per_prompt": 64}


def tokenize(tokenizer, row):
    prefix = tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    answer = json.dumps(row["action"], separators=(",", ":")) + tokenizer.eos_token
    inputs = tokenizer(prefix + answer, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    if inputs[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt/answer token boundary is not stable")
    if len(inputs) > LIMITS["max_sequence_tokens"]:
        raise ValueError("Example exceeds fixed sequence cap; no silent truncation")
    return {"input_ids": inputs, "labels": [-100] * len(prompt_ids) + inputs[len(prompt_ids):],
            "prompt_tokens": len(prompt_ids), "action_tokens": len(inputs) - len(prompt_ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=SMOKE / "outputs/sft")
    parser.add_argument("--preflight-only", action="store_true", help="Tokenizer/import check; no model or CUDA load")
    args = parser.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    train_rows = [json.loads(line) for line in (SMOKE / "data/train.jsonl").read_text().splitlines()]
    valid_rows = [json.loads(line) for line in (SMOKE / "data/validation.jsonl").read_text().splitlines()]
    samples = [tokenize(tokenizer, row) for row in train_rows]
    expected_tokens = sum(len(sample["input_ids"]) for sample in samples)
    if len(samples) != LIMITS["microbatches"]:
        raise ValueError("Training example count differs from fixed microbatch cap")
    if expected_tokens > LIMITS["seen_token_cap"]:
        raise ValueError("Planned training tokens exceed fixed seen-token cap")
    result = {"purpose": "tiny SFT workflow/throughput smoke only", "limits": LIMITS,
              "data_sha256": sha256(SMOKE / "data/train.jsonl"),
              "model_path": str(args.model), "thinking_enabled": False,
              "versions": versions(["torch", "transformers", "peft", "accelerate", "huggingface-hub"]),
              "unique_examples": len(samples), "planned_seen_tokens": expected_tokens,
              "max_example_tokens": max(len(sample["input_ids"]) for sample in samples),
              "model_provenance": json.loads((args.model.parent / "qwen_manifest.json").read_text())}
    if result["model_provenance"].get("status") != "verified_complete":
        raise ValueError("Official model download has not been fully verified")
    for file in result["model_provenance"]["files"]:
        if sha256(args.model / file["path"]) != file["sha256"]:
            raise ValueError("Official model file changed: " + file["path"])
    if args.preflight_only:
        result.update(status="tokenizer_preflight_passed", GPU_work_performed=False,
                      imports_checked=["AutoTokenizer", "AutoModelForCausalLM", "LoraConfig", "get_peft_model",
                                       "PeftModel", "get_peft_model_state_dict"])
        save_json(args.output / "preflight.json", result)
        print(json.dumps({key: result[key] for key in ["status", "unique_examples", "planned_seen_tokens", "max_example_tokens"]}))
        return
    import torch
    from transformers import set_seed
    result["runtime"] = require_cuda_job(torch)
    budget = Budget(LIMITS["train_seconds"])
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    seen_tokens = 0
    action_tokens = 0
    optimizer_steps = 0
    t0 = time.monotonic()
    try:
        set_seed(20261023)
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                               target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        torch.cuda.synchronize()
        result["load_seconds"] = time.monotonic() - t0
        result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer.zero_grad(set_to_none=True)
        for micro, sample in enumerate(samples):
            budget.check()
            if seen_tokens + len(sample["input_ids"]) > LIMITS["seen_token_cap"]:
                raise RuntimeError("Seen-token cap reached")
            torch.cuda.synchronize()
            start = time.monotonic()
            inputs = torch.tensor([sample["input_ids"]], device="cuda")
            labels = torch.tensor([sample["labels"]], device="cuda")
            output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels)
            if not torch.isfinite(output.loss):
                raise RuntimeError("Non-finite training loss")
            (output.loss / LIMITS["gradient_accumulation"]).backward()
            gradient_norm = None
            if (micro + 1) % LIMITS["gradient_accumulation"] == 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise RuntimeError("Non-finite training gradient norm")
                gradient_norm = float(norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            torch.cuda.synchronize()
            seconds = time.monotonic() - start
            seen_tokens += len(sample["input_ids"])
            action_tokens += sample["action_tokens"]
            rec = {"microbatch": micro + 1, "optimizer_steps": optimizer_steps,
                   "loss": float(output.loss.detach()), "seconds": seconds,
                   "seen_tokens": seen_tokens, "action_tokens": action_tokens,
                   "gradient_norm_before_clip": gradient_norm, **resources(torch)}
            records.append(rec)
            with (args.output / "steps.jsonl").open("a") as handle:
                handle.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            del inputs, labels, output
        budget.check()
        adapter_path = args.output / "adapter"
        model.save_pretrained(adapter_path, safe_serialization=True)
        tokenizer.save_pretrained(adapter_path)
        expected_adapter_state = {key: value.detach().cpu().clone()
                                  for key, value in get_peft_model_state_dict(model).items()}
        model.gradient_checkpointing_disable()
        del optimizer
        budget.check()
        torch.cuda.synchronize()
        reload_start = time.monotonic()
        # unload() removes LoRA without merging it into the frozen base weights.
        base_model = model.unload()
        del model
        if any("lora_" in name for name, _ in base_model.named_parameters()):
            raise RuntimeError("LoRA parameters remain after adapter unload")
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False,
                                          local_files_only=True)
        reloaded_state = get_peft_model_state_dict(model)
        if set(reloaded_state) != set(expected_adapter_state):
            raise RuntimeError("Reloaded adapter parameter keys differ from saved adapter")
        if not all(torch.equal(reloaded_state[key].detach().cpu(), expected_adapter_state[key])
                   for key in expected_adapter_state):
            raise RuntimeError("Reloaded adapter parameters differ from saved adapter")
        torch.cuda.synchronize()
        result["adapter_reload"] = {"unloaded_without_merge": True, "reloaded_from_disk": True,
            "state_exact_match": True, "parameter_tensors": len(reloaded_state),
            "seconds": time.monotonic() - reload_start,
            "safetensors_sha256": sha256(adapter_path / "adapter_model.safetensors")}
        del expected_adapter_state, reloaded_state, base_model
        budget.check()
        model.eval()
        model.config.use_cache = True
        generations = []
        for row in valid_rows[:LIMITS["validation_prompts"]]:
            budget.check()
            inputs = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to("cuda")
            if inputs.shape[-1] > LIMITS["max_sequence_tokens"]:
                raise ValueError("Validation prompt too long")
            torch.cuda.synchronize()
            start = time.monotonic()
            with torch.no_grad():
                output = model.generate(inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                    max_new_tokens=LIMITS["generation_tokens_per_prompt"], pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            text = tokenizer.decode(output[0, inputs.shape[-1]:], skip_special_tokens=True).strip()
            try:
                action = json.loads(text)
                if set(action) != {"op", "sites"}:
                    raise ValueError("Action JSON contains extra/missing keys")
                i, j = action["sites"]
                if type(i) is not int or type(j) is not int or i == j or min(i, j) < 0 or max(i, j) >= 324:
                    raise ValueError("Invalid action site IDs")
                expected = {"swap_cations": {39, 40}, "move_vacancy": {0, 8}}
                if {row["numbers"][i], row["numbers"][j]} != expected[action["op"]]:
                    raise ValueError("Action violates sublattice occupancy")
                legal = True
            except (ValueError, TypeError, KeyError, IndexError):
                legal = False
            generations.append({"id": row["id"], "text": text, "legal_json": legal,
                                "new_tokens": int(output.shape[-1] - inputs.shape[-1]),
                                "seconds": time.monotonic() - start})
        save_json(args.output / "generations.json", generations)
        if optimizer_steps != LIMITS["optimization_steps"]:
            raise RuntimeError("Completed optimization steps differ from fixed step target")
        if seen_tokens > LIMITS["seen_token_cap"]:
            raise RuntimeError("Actual training tokens exceed fixed seen-token cap")
        result.update(status="completed", optimization_steps=optimizer_steps,
                      validation_legal_json=sum(g["legal_json"] for g in generations),
                      validation_n=len(generations), generation_tokens=sum(g["new_tokens"] for g in generations))
    except Exception as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        steady = [r["seconds"] for r in records[4:]]
        result.update(seen_tokens=seen_tokens, action_tokens=action_tokens, microbatches=len(records),
                      optimizer_steps=optimizer_steps, wall_seconds=time.monotonic() - t0,
                      first_four_microbatch_seconds=[r["seconds"] for r in records[:4]],
                      steady_microbatches=distribution(steady), resources=resources(torch))
        if records:
            result["training_tokens_per_second"] = seen_tokens / sum(r["seconds"] for r in records)
        save_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
