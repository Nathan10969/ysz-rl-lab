#!/usr/bin/env python3
"""Recompute saved YSZ metrics, without loading any model or contacting a cluster."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = (
    ("meta_s0b_eval", "SFT v2", "旧表示"),
    ("meta_s3_eval", "GRPO 元流程第 3 步", "旧表示"),
    ("meta_s6_greedy", "局部特征贪心", "候选菜单，非 LLM"),
    ("meta_s6_menurandom", "同菜单随机", "候选菜单，主要随机对照"),
    ("meta_s6_sft_eval", "SFT v4", "候选菜单"),
    ("meta_s6_eval", "GRPO v4", "候选菜单"),
)


def load(name: str) -> dict:
    if not name or Path(name).name != name:
        raise ValueError("Pass a job name, not a filesystem path")
    return json.loads((ROOT / "experiments/results" / (name + ".json")).read_text())


def values(record: dict) -> np.ndarray:
    if record.get("status") != "COMPLETED":
        raise ValueError("Only complete jobs can be compared as finished evaluations")
    rows = record["per_start"]
    if not rows:
        raise ValueError("No per-start evaluation data")
    return np.array([r["drop_eV"] for r in rows], dtype=float)


def mean_se(record: dict) -> tuple[float, float]:
    data = values(record)
    rng = np.random.default_rng(0)
    boot = [rng.choice(data, len(data)).mean() for _ in range(2000)]
    return float(data.mean()), float(np.std(boot))


def paired(candidate: dict, reference: dict) -> dict:
    values(candidate)
    values(reference)
    if candidate.get("seed") != reference.get("seed"):
        raise ValueError("Different seeds cannot be treated as paired starts")
    if candidate.get("config", {}).get("pool") != reference.get("config", {}).get("pool"):
        raise ValueError("Different start pools cannot be compared as paired evaluations")
    if candidate.get("protocol") != reference.get("protocol"):
        raise ValueError("Different physical protocols need a separate comparison")
    a = {row["start_index"]: row for row in candidate["per_start"]}
    b = {row["start_index"]: row for row in reference["per_start"]}
    if len(a) != len(candidate["per_start"]) or len(b) != len(reference["per_start"]):
        raise ValueError("Duplicate start identifiers")
    if a.keys() != b.keys():
        raise ValueError("Different start identifiers")
    keys = sorted(a)
    if any(abs(a[k]["initial_energy_eV"] - b[k]["initial_energy_eV"]) > 1e-6 for k in keys):
        raise ValueError("Initial energies differ; start alignment has not been verified")
    delta = np.array([a[k]["drop_eV"] - b[k]["drop_eV"] for k in keys])
    rng = np.random.default_rng(1)
    boot = np.array([rng.choice(delta, len(delta)).mean() for _ in range(4000)])
    se = float(boot.std())
    return {"candidate": candidate["job"], "reference": reference["job"], "n": len(keys),
            "mean_improvement_eV": float(delta.mean()), "paired_se_eV": se,
            "bootstrap_percentile_95_eV": np.quantile(boot, [.025, .975]).tolist(),
            "exploratory_acceptance_threshold_eV": max(.1, se),
            "exploratory_rule_pass": bool(delta.mean() >= max(.1, se)),
            "interpretation": "Exploratory paired comparison; repeated validation and multiple comparisons require independent confirmation."}


def benchmark_table() -> str:
    lines = ["| 方法 | 平均降能 ± 标准误差（eV） | 起点数 | 设置 |", "|---|---:|---:|---|"]
    for job, label, setting in BENCHMARKS:
        path = ROOT / "experiments/results" / (job + ".json")
        if not path.exists():
            lines.append(f"| {label} | 尚无完整本地结果 | — | {setting} |")
            continue
        record = load(job)
        if record.get("status") != "COMPLETED":
            lines.append(f"| {label} | {record.get('status', 'unknown')} | — | {setting} |")
            continue
        mean, se = mean_se(record)
        lines.append(f"| {label} | {mean:.3f} ± {se:.3f} | {len(record['per_start'])} | {setting} |")
    return "\n".join(lines)


def write_reports():
    table = benchmark_table()
    readme = ROOT / "README.md"
    text = readme.read_text()
    begin, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    before, remainder = text.split(begin, 1)
    _, after = remainder.split(end, 1)
    readme.write_text(before + begin + "\n" + table + "\n" + end + after)
    lines = ["# 保存结果看板", "", "由 `python tools/results.py --write` 从公开的逐起点摘要生成。",
             "单位：MACE eV／316 原子超胞；每个起点取四条轨迹中的最好降幅，再求平均。± 为起点重采样标准误差。",
             "这是已保存结果的复算，不会调用模型或提交作业。", "", table, "", "## 配对比较", "",
             "预先记录的探索门槛为 `max(0.10 eV, 1 × 配对标准误差)`。通过探索门槛不等价于独立确认。",
             "旧随机游走走另一套搜索代码路径；当前主要随机对照是同框架 menurandom。", "",
             "| 候选相对对照 | 平均差（eV） | 配对 SE | 探索规则 |", "|---|---:|---:|---|"]
    for candidate, reference in (("meta_s3_eval", "meta_s0b_eval"), ("meta_s3_eval", "meta_s6_menurandom"),
                                  ("meta_s6_greedy", "meta_s6_menurandom"), ("meta_s6_sft_eval", "meta_s6_menurandom"),
                                  ("meta_s6_sft_eval", "meta_s6_greedy"), ("meta_s6_eval", "meta_s6_menurandom"),
                                  ("meta_s6_eval", "meta_s6_greedy")):
        try:
            comparison = paired(load(candidate), load(reference))
        except FileNotFoundError:
            continue
        except ValueError:
            continue
        mark = "达到" if comparison["exploratory_rule_pass"] else "未达到"
        lines.append(f"| {candidate} − {reference} | {comparison['mean_improvement_eV']:+.3f} | {comparison['paired_se_eV']:.3f} | {mark} |")
    lines += ["", "## 当前判断", "", "- 两参数特征贪心在当前验证池上优于旧表示的模型和随机选择器。",
              "- 菜单 SFT 没有达到同时超越随机和贪心的门槛。",
              "- 菜单 GRPO 的状态、是否通过两个对照，以本页表格与保存结果为准；未独立确认前不写成最终结论。",
              "- 当前验证池被反复用于选择方案，后续确认应使用未参与选择的起点和多个种子。", "",
              "## 全部已保存作业", "", "这一索引包含采集、训练、rollout 和验证等不同阶段；不能把训练中的收益当独立测试表现。", "",
              "| 作业 | 阶段 | 状态 | 起点数 | 平均降能（eV） |", "|---|---|---|---:|---:|"]
    for file in sorted((ROOT / "experiments/results").glob("*.json")):
        if file.stem == "index":
            continue
        item = json.loads(file.read_text())
        mean = item["mean_best_of_4_drop_eV"]
        score = "—" if mean is None else f"{mean:.6f}"
        lines.append(f"| [{item['job']}](../../experiments/results/{file.name}) | {item.get('stage', '—')} | {item.get('status', '—')} | {len(item['per_start'])} | {score} |")
    lines += ["", "来源和变换见 [清单](../../provenance/manifest.json)。历史解释及更正见 [元流程日志](../../history/formal12/META_LOG.md)。", ""]
    (ROOT / "docs/experiments/results.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", nargs=2, metavar=("CANDIDATE", "REFERENCE"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_reports()
    elif args.paired:
        print(json.dumps(paired(load(args.paired[0]), load(args.paired[1])), ensure_ascii=False, indent=2))
    else:
        print(benchmark_table())


if __name__ == "__main__":
    main()
