#!/usr/bin/env python3
"""Export an explicit, sanitized research snapshot. Reads local files; never runs research jobs."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ZONE = ZoneInfo("America/Los_Angeles")
TOKEN_PATTERNS = (
    r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
    r"\bgithub_pat_[A-Za-z0-9_]{30,}\b",
    r"\bsk-(?:proj-)?[A-Za-z0-9_-]{25,}\b",
    r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----",
    r"\bAKIA[A-Z0-9]{16}\b",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean(text: str) -> str:
    """Replace machine/user identifiers while retaining relative paths and scientific numbers."""
    text = re.sub(r"/Users/[^/\s\"'`<>]+", "/LOCAL_HOME", text)
    text = re.sub(r"~/\.rsi_sealed/[^\s\"'`<>]+", "SEALED_SEED_FILE", text)
    text = re.sub(r"/data/home/[^/\s\"'`<>]+/[^/\s\"'`<>]+/(?:Jony/)?", "/CLUSTER_HOME/", text)
    text = re.sub(r"\byg\d{5,}\b", "CLUSTER_USER", text)
    text = re.sub(r"\b[A-Za-z][A-Za-z0-9_-]*-a100\b", "your-cluster", text)
    text = re.sub(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", "CLUSTER_HOST", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "researcher@example.org", text)
    for pattern in TOKEN_PATTERNS:
        if re.search(pattern, text):
            raise ValueError("Credential-like content found; export stopped without printing its value")
    return text


def encoded(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def read_stable(path: Path) -> bytes:
    for _ in range(3):
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
        if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
            return content
    raise RuntimeError("Source changed during snapshot; retry after the writer finishes: " + path.name)


def compact_result(name: str, result: dict) -> dict:
    per_start = []
    if "groups" in result:
        for group in result["groups"]:
            best = group["best_energy_eV"]
            best = min(best) if isinstance(best, list) else best
            per_start.append({"start_index": group["group"], "initial_energy_eV": group["start_energy_eV"],
                              "best_energy_eV": best, "drop_eV": group["start_energy_eV"] - best})
    elif "chains" in result:
        grouped = {}
        for chain in result["chains"]:
            if not {"start", "initial_energy_eV", "best_energy_eV"}.issubset(chain):
                continue
            grouped.setdefault(chain["start"], []).append(chain)
        for start, chains in sorted(grouped.items()):
            best = max(chains, key=lambda c: c["initial_energy_eV"] - c["best_energy_eV"])
            per_start.append({"start_index": start, "initial_energy_eV": best["initial_energy_eV"],
                              "best_energy_eV": best["best_energy_eV"],
                              "drop_eV": best["initial_energy_eV"] - best["best_energy_eV"]})
    record = {"job": name}
    for key in ("schema", "status", "arm", "stage", "seed", "round", "protocol", "protocol_sha", "config",
                "counters", "limits", "wall_seconds", "constrained_diagnostics", "replaced_starts"):
        if key in result:
            record[key] = result[key]
    if result.get("updates"):
        record["optimizer_updates_recorded"] = len(result["updates"])
    if result.get("schema") == "formal12-imitate-v1":
        metric_keys = ("n", "captured_fraction", "argmax_accuracy", "optimal_mass", "chance_mass",
                       "expected_dNN", "menu_min_dNN")
        record["menu_preference"] = {
            name: {split: {k: metrics[k] for k in metric_keys if k in metrics}
                   for split, metrics in splits.items() if split in {"test", "train_fit"}}
            for name, splits in result.get("eval", {}).items()}
        record["diagnostic_data"] = {k: result["data"][k] for k in
            ("train_rows", "test_rows", "train_fit_rows", "mean_optimal_set_size")
            if k in result.get("data", {})}
        record["diagnostic_scope"] = (
            "Menu preference on development data; C is a frozen-heuristic score capture, "
            "not physical energy improvement or a held-out final test.")
    record["per_start"] = per_start
    record["mean_best_of_4_drop_eV"] = statistics.mean(x["drop_eV"] for x in per_start) if per_start else None
    record["scope"] = "Derived from saved results. MACE eV per supercell; not final DFT validation."
    return json.loads(clean(json.dumps(record, ensure_ascii=False, allow_nan=False)))


def budget_summary(source: Path) -> tuple[dict, dict]:
    out, hashes = {}, {}
    for filename in ("FORMAL_TRAINING_POOL.json", "FORMAL_DFT_DIAG.json", "FORMAL_MD_POOL.json", "BUDGET.json"):
        path = source / "training_smoke" / filename
        if not path.exists():
            continue
        raw = read_stable(path)
        hashes[str(path.relative_to(source))] = sha(raw)
        data = json.loads(raw)
        summary = {k: data[k] for k in ("schema", "authorized_card_hours", "actual_card_hours", "reserved_card_hours",
                   "actual_plus_reserved", "updated_at_utc", "pool_actual_before_card_hours", "diag_cap_card_hours") if k in data}
        jobs = data.get("sacct_jobs", data.get("jobs", []))
        summary["jobs"] = [{k: j[k] for k in ("name", "slurm_job_id", "state", "elapsed_seconds", "actual_card_hours",
                           "card_hours", "hard_card_hours", "terminal") if k in j} for j in jobs]
        summary["note"] = "Snapshot of the local accounting ledger; missing fields are not assumed to be zero."
        if all(isinstance(data.get(k), (int, float)) for k in
               ("actual_card_hours", "reserved_card_hours", "actual_plus_reserved")):
            summary["header_sum_card_hours"] = data["actual_card_hours"] + data["reserved_card_hours"]
            summary["header_totals_consistent"] = math.isclose(
                summary["header_sum_card_hours"], data["actual_plus_reserved"], abs_tol=1e-8)
            if not summary["header_totals_consistent"]:
                summary["note"] += (
                    " Header totals disagree. Original values are preserved; reconcile the source ledger "
                    "before treating them as current settled spending or remaining capacity.")
        out[filename] = summary
    return out, hashes


def build(source: Path, date: str) -> bool:
    source = source.resolve()
    if not (source / "PROJECT_RECORD_RSI_YSZ.md").is_file():
        raise ValueError("Source is not the expected YSZ research workspace")
    config = json.loads((ROOT / "tools/publication_sources.json").read_text())
    outputs, input_hashes, mappings = {}, {}, []
    for item in config["files"]:
        rel, dest = item["source"], item["destination"]
        path = (source / rel).resolve()
        path.relative_to(source)
        if not path.is_file():
            raise FileNotFoundError("Allowlisted source is missing: " + rel)
        raw = read_stable(path)
        public = clean(raw.decode("utf-8")).encode()
        outputs[dest] = public
        input_hashes[rel] = sha(raw)
        mappings.append({"source": rel, "published": dest, "source_sha256": sha(raw), "published_sha256": sha(public)})
    results = []
    for path in sorted((source / "formal12/jobs").glob("*/out/result.json")):
        raw = read_stable(path)
        item = compact_result(path.parents[1].name, json.loads(raw))
        dest = "experiments/results/" + item["job"] + ".json"
        public = encoded(item)
        outputs[dest] = public
        rel = str(path.relative_to(source))
        input_hashes[rel] = sha(raw)
        mappings.append({"source": rel, "published": dest, "source_sha256": sha(raw), "published_sha256": sha(public),
                         "transformation": "selected metadata, counters, and per-start energy drops; no weights or trajectories"})
        results.append(item)
    budgets, budget_hashes = budget_summary(source)
    input_hashes.update(budget_hashes)
    previous_path = ROOT / "provenance/manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    # Timestamp-only accounting refreshes must not manufacture a daily update.
    semantic_budgets = {name: {k: v for k, v in data.items() if k != "updated_at_utc"}
                        for name, data in budgets.items()}
    fingerprint = sha(encoded({"inputs": {k: v for k, v in input_hashes.items() if k not in budget_hashes},
                              "budgets": semantic_budgets, "allowlist": config,
                              "exporter_sha256": sha(Path(__file__).read_bytes())}))
    if previous.get("input_fingerprint") == fingerprint:
        print("No source changes; public snapshot left unchanged")
        return False
    previous_results = {}
    for entry in previous.get("files", []):
        if entry["published"].startswith("experiments/results/"):
            previous_results[entry["published"]] = entry["source_sha256"]
    changed = [m for m in mappings if m["published"].startswith("experiments/results/") and
               previous_results.get(m["published"]) != m["source_sha256"]]
    outputs["experiments/results/index.json"] = encoded([
        {k: r.get(k) for k in ("job", "status", "stage", "arm", "seed", "round", "mean_best_of_4_drop_eV")}
        for r in results])
    outputs["experiments/budgets.json"] = encoded(budgets)
    outputs["provenance/manifest.json"] = encoded({"schema": "ysz-public-snapshot-v1", "snapshot_date_local": date,
        "timezone": "America/Los_Angeles", "input_fingerprint": fingerprint, "files": mappings,
        "ledger_source_sha256": budget_hashes, "redactions": ["local user paths", "cluster account paths",
        "cluster host aliases", "IPv4 literals", "email addresses"], "excluded": ["model weights", "VASP potentials",
        "third-party papers and vendored source", "virtual environments", "credentials", "private message stores",
        "raw atom coordinates and per-forward force logs"],
        "history_note": "Historical files are retrospective source snapshots, not backdated Git commits."
    })
    report = ["# 当日同步记录 · " + date, "", "本页记录本次从本机已有文件同步到的证据。日期是同步日期，不冒充作业执行日期或个人学习完成日期。",
              "", "## 新增或变化的结果", "", "| 作业 | 状态 | 验证起点数 | 平均降能（eV/超胞） |", "|---|---|---:|---:|"]
    by_name = {r["job"]: r for r in results}
    for mapping in changed:
        name = Path(mapping["published"]).stem
        item = by_name[name]
        score = item["mean_best_of_4_drop_eV"]
        report.append(f"| [{name}](../{mapping['published']}) | {item.get('status', 'unknown')} | {len(item['per_start'])} | {score:.6f} |"
                      if score is not None else f"| [{name}](../{mapping['published']}) | {item.get('status', 'unknown')} | — | — |")
    if not changed:
        report += ["", "本次没有新的结果文件；更新来自研究记录、代码或本机账本。"]
    report += ["", "## 判读与学习记录", "", "自动导出只列事实。实验含义、与既有结论的冲突及当天亲自理解的内容，由维护者核对后补充。",
               "已有学习轨迹见 [项目学习笔记](../docs/learning/README.md)，历史过程见 [历史索引](../history/README.md)。", ""]
    journal_path = ROOT / ("journal/" + date + "-sync.md")
    if journal_path.exists():
        # Keep already published observations for the day; new observations append, never erase.
        old = journal_path.read_text()
        report = [old.rstrip(), "", "---", "", "## 同日后续同步", ""] + report[4:]
    outputs[str(journal_path.relative_to(ROOT))] = ("\n".join(report).rstrip() + "\n").encode()
    journal_names = {p.name for p in (ROOT / "journal").glob("*.md")
                     if p.name not in {"README.md", "TEMPLATE.md"}}
    journal_names.add(journal_path.name)
    journal_index = ["# 每日记录", "", "研究历史从 2026-10-22 开始；本仓库按真实同步日期记录，不回填过去的 Git 提交。", ""]
    for name in sorted(journal_names, reverse=True):
        journal_index.append(f"- [{name}]({name})")
    journal_index += ["", "`-sync.md` 是已有证据的同步清单，其它日期记录用于维护者补充实验解释和实际学习收获。",
                      "[记录模板](TEMPLATE.md) · [完整实验时间线](../docs/experiments/timeline.md)", ""]
    outputs["journal/README.md"] = "\n".join(journal_index).encode()
    links = ["# 历史材料索引", "", "这些是研究过程中形成的原有记录的脱敏副本。旧判断、作业状态及资源授权只代表各文件写作时的情况。",
             "历史文档里的原工作区相对路径可结合 [来源清单](../provenance/manifest.json) 查找；未公开的原始资产链接不能在公开仓库直接打开。",
             "当前判读见 [项目首页](../README.md) 和 [实验结果](../docs/experiments/results.md)。", "", "## 计划、审核、结果和归档版本", ""]
    for m in mappings:
        if m["published"].startswith("history/"):
            links.append(f"- [{m['source']}]({m['published'].removeprefix('history/')})")
    outputs["history/README.md"] = ("\n".join(links) + "\n").encode()
    # Prepare and validate all payloads before any public write.
    for dest, payload in outputs.items():
        target = (ROOT / dest).resolve()
        target.relative_to(ROOT.resolve())
        clean(payload.decode())
        if len(payload) > 5_000_000:
            raise ValueError("Unexpected large public file: " + dest)
    for dest, payload in outputs.items():
        target = ROOT / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".publishing")
        temporary.write_bytes(payload)
        temporary.replace(target)
    print(json.dumps({"changed": True, "files_exported": len(mappings), "result_files": len(results),
                      "changed_result_files": len(changed), "input_fingerprint": fingerprint}, ensure_ascii=False))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--date", default=datetime.now(LOCAL_ZONE).date().isoformat())
    args = parser.parse_args()
    source = args.source_root
    if source is None:
        local = ROOT / ".local-source.json"
        if not local.exists():
            parser.error("Pass --source-root or configure the git-ignored .local-source.json")
        source = Path(json.loads(local.read_text())["source_root"])
    datetime.strptime(args.date, "%Y-%m-%d")
    build(source, args.date)


if __name__ == "__main__":
    main()
