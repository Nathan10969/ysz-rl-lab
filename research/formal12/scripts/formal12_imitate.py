"""Step-6 follow-up (2026-10-27): can the menu policy learn the frozen greedy rule when the labels carry it?

No physics. Menus (make_menu, 16 legal candidates with dNN/dNNN) are built on existing D0 states; the label is the
frozen greedy choice (GREEDY_WEIGHTS, fitted on seed-1 D0 only). Training uses the SFT_v4 recipe unchanged
(sft_from_base, same LoRA, lr, epochs, copies) so only the labels differ. Held-out evaluation uses D0 states of
another seed; the exact constrained choice distribution over the menu is computed per prompt (no sampling noise).
A pass only shows that the learning pathway can represent the rule; it is not evidence of value beyond greedy.
--eval-only (2026-10-27, step-GRPO readout): no training; the same test menus (same seed and rows) for given adapters.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback

import numpy as np

import claude08_env as env
from formal12_common import LIMITS, Ledger, Phase, save_json, source_hashes, runtime_info
from formal12_menu import GREEDY_WEIGHTS, MenuTrie, greedy_choice, make_menu, to_v4

SCHEMA = 'formal12-imitate-v1'


def greedy_score(m):
    return GREEDY_WEIGHTS[0] * m['dNN'] + GREEDY_WEIGHTS[1] * m['dNNN']


def build_rows(rows_v2, *, seed, copies, label):
    """D0 states -> menu rows labelled with the frozen greedy choice (random tie-break, as greedy_choice)."""
    out = []
    for index, row in enumerate(rows_v2):
        observation = json.loads(row['messages'][1]['content'])
        occupancy = env.decode_occupancy(observation['occupancy'])
        for copy in range(copies):
            menu = make_menu(occupancy, (seed, 'imitate', label, index, copy))
            scores = [greedy_score(m) for m in menu]
            best = min(scores)
            target = greedy_choice(menu, np.random.default_rng([seed, index, copy]))['k']
            out.append(dict(messages=to_v4(row['messages'], menu), choice=target, menu=menu, scores=scores,
                            optimal=[m['k'] for m, s in zip(menu, scores) if abs(s - best) < 1e-12]))
    return out


def choice_distribution(model, tokenizer, messages, menu, torch, device, ledger):
    """Exact constrained log-probabilities of every {"choice":k} (the rollout sampling distribution).
    Each forward is charged before it runs, prompt plus branch prefix (2026-10-27 fix after Codex review)."""
    trie = MenuTrie(tokenizer, menu)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if hasattr(ids, 'tolist'): ids = ids.tolist()
    branch = {}
    with torch.no_grad():
        for prefix, allowed in trie.children.items():
            if len(allowed) == 1: continue
            ledger.before_prefill(len(ids) + len(prefix), phase='imitate_eval')
            logits = model(input_ids=torch.tensor([ids + list(prefix)], device=device)).logits[0, -1].float()
            if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite logits')
            branch[prefix] = dict(zip(allowed, logits.index_select(0, torch.tensor(allowed, device=device)).log_softmax(-1).tolist()))
    logp = {}
    for sequence, m in trie.leaves.items():
        logp[m['k']] = sum(branch[sequence[:i]][sequence[i]] for i in range(len(sequence)) if sequence[:i] in branch)
    return [logp[m['k']] for m in menu]


def evaluate(model, tokenizer, rows, ledger, torch, device):
    hits, mass, chance, gained, possible, chosen_dNN, menu_min_dNN = [], [], [], [], [], [], []
    for row in rows:
        ledger.check()
        lp = choice_distribution(model, tokenizer, row['messages'], row['menu'], torch, device, ledger)
        p = np.exp(np.array(lp)); p /= p.sum()
        scores = np.array(row['scores']); optimal = set(row['optimal'])
        hits.append(int(np.argmax(p)) in optimal)
        mass.append(float(sum(p[k] for k in optimal)))
        chance.append(len(optimal) / len(scores))
        gained.append(float(scores.mean() - p @ scores)); possible.append(float(scores.mean() - scores.min()))
        dNN = np.array([m['dNN'] for m in row['menu']])
        chosen_dNN.append(float(p @ dNN)); menu_min_dNN.append(float(dNN.min()))
    return dict(n=len(rows), argmax_accuracy=float(np.mean(hits)), optimal_mass=float(np.mean(mass)),
                chance_mass=float(np.mean(chance)), captured_fraction=float(np.sum(gained) / np.sum(possible)),
                expected_dNN=float(np.mean(chosen_dNN)), menu_min_dNN=float(np.mean(menu_min_dNN)))


def adapter_name(adapter):
    """jobs/<name>/out/... -> <name> (also for policy_adapter/policy paths)."""
    parts = Path(adapter).parts
    return parts[parts.index('jobs') + 1] if 'jobs' in parts else Path(adapter).parent.parent.name


def load_policy(model_path, adapter, device, dtype, torch):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                                    dtype=getattr(torch, dtype), attn_implementation='sdpa')
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
    finally:
        torch.set_default_dtype(previous)
    return model.to(device).eval()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=2026102301)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--qwen-manifest', type=Path, required=True)
    parser.add_argument('--train-rows', type=Path, help='seed-1 D0 rows (same source as SFT_v4); not with --eval-only')
    parser.add_argument('--test-rows', type=Path, required=True, help='held-out D0 rows of another seed')
    parser.add_argument('--compare-adapter', type=Path, nargs='*', default=[], help='e.g. the SFT_v4 adapter')
    parser.add_argument('--copies', type=int, default=4)
    parser.add_argument('--sft-epochs', type=int, default=3)
    parser.add_argument('--train-fit-rows', type=int, default=64)
    parser.add_argument('--eval-only', action='store_true', help='no training: evaluate --compare-adapter on the test menus')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test-limit', type=int, default=None, help='CPU tests only: first N train and test states')
    args = parser.parse_args(argv)
    if args.test_limit is not None and args.device != 'cpu': raise ValueError('--test-limit is CPU-test only')
    if args.eval_only == (args.train_rows is not None): raise ValueError('--train-rows is required unless --eval-only')
    if args.eval_only and not args.compare_adapter: raise ValueError('--eval-only needs --compare-adapter')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    ledger = Ledger(replace(LIMITS, max_relax_actual=0), log_path=args.output / 'cost_events.jsonl')
    result = dict(schema=SCHEMA, status='STARTED', source_hashes=source_hashes(), greedy_weights=list(GREEDY_WEIGHTS),
                  inputs={k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v)
                          for k, v in vars(args).items()})
    try:
        result['runtime'] = runtime_info(args.device)
        import torch
        from transformers import AutoTokenizer
        from formal12_actor import sft_from_base, verify_qwen
        test_v2 = json.loads(args.test_rows.read_text())
        train_v2 = [] if args.eval_only else json.loads(args.train_rows.read_text())
        if args.test_limit: train_v2, test_v2 = train_v2[:args.test_limit], test_v2[:args.test_limit]
        train = build_rows(train_v2, seed=args.seed, copies=args.copies, label='train')
        test = build_rows(test_v2, seed=args.seed, copies=1, label='test')
        fit = train[::args.copies][:args.train_fit_rows]
        result['data'] = dict(train_rows=len(train), test_rows=len(test), train_fit_rows=len(fit),
                              target_histogram=np.bincount(np.array([r['choice'] for r in train], dtype=int), minlength=16).tolist(),
                              mean_optimal_set_size=float(np.mean([len(r['optimal']) for r in test])))
        save_json(args.output / 'rows.json', [dict(messages=r['messages'], choice=r['choice'], optimal=r['optimal'],
                                                   scores=r['scores']) for r in train + test])
        if not args.eval_only:
            with Phase(ledger, 'imitate_sft', rows=len(train)):
                result['sft'] = sft_from_base(args.model, args.qwen_manifest, train, args.output, ledger, device=args.device,
                                              epochs=args.sft_epochs, seed=args.seed)
        verify_qwen(args.model, args.qwen_manifest)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        dtype = 'bfloat16' if args.device == 'cuda' else 'float32'
        result['eval'] = {}
        own = [] if args.eval_only else [('greedy_sft', args.output / 'adapter')]
        for name, adapter in own + [(adapter_name(a), a) for a in args.compare_adapter]:
            model = load_policy(args.model, adapter, args.device, dtype, torch)
            with Phase(ledger, 'imitate_eval', adapter=name):
                result['eval'][name] = dict(test=evaluate(model, tokenizer, test, ledger, torch, args.device))
                if fit: result['eval'][name]['train_fit'] = evaluate(model, tokenizer, fit, ledger, torch, args.device)
            del model
            if args.device == 'cuda': torch.cuda.empty_cache()
        result['status'] = 'COMPLETED'
    except env.BudgetExceeded as exc:
        result.update(status='BUDGET_EXHAUSTED', error=str(exc), traceback=traceback.format_exc())
    except Exception as exc:
        result.update(status='FAILED', error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
    finally:
        from dataclasses import asdict
        result.update(counters=asdict(ledger.counters), wall_seconds=ledger.elapsed(), phase_timings=ledger.phase_timings)
        save_json(args.output / 'result.json', result)
    print(json.dumps({k: result.get(k) for k in ('status', 'data', 'eval', 'wall_seconds', 'error')}, default=str)[:3000])
    if result['status'] != 'COMPLETED': raise SystemExit(1)


if __name__ == '__main__': main()
