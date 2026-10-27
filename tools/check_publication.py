#!/usr/bin/env python3
"""Check the public package. No model loads, shell commands, network, or research writes."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from publish_snapshot import TOKEN_PATTERNS


def public_files():
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(part in {".git", ".venv", "venv", "__pycache__", ".local", "node_modules"} for part in rel.parts):
            continue
        if path.is_file() and path.name not in {".local-source.json", ".DS_Store"}:
            yield path


def check() -> dict:
    errors, files, python_files = [], list(public_files()), 0
    for path in files:
        rel = str(path.relative_to(ROOT))
        if path.is_symlink():
            errors.append("Symlink is outside the publication scope: " + rel)
            continue
        if path.name.startswith((".env", "POTCAR", "WAVECAR", "CHGCAR", "OUTCAR")) or path.suffix in {
            ".safetensors", ".pt", ".pth", ".model", ".ckpt", ".pdf", ".pem", ".key"}:
            errors.append("Private or binary asset in public tree: " + rel)
            continue
        if path.stat().st_size > 5_000_000:
            errors.append("Unexpected large file: " + rel)
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            errors.append("Unexpected binary file: " + rel)
            continue
        for pattern in TOKEN_PATTERNS:
            if re.search(pattern, text):
                errors.append("Credential-like content in " + rel)
        for pattern in (r"/Users/[A-Za-z0-9_.-]+/", r"\byg\d{5,}\b", r"\b[A-Za-z][A-Za-z0-9_-]*-a100\b"):
            if re.search(pattern, text):
                errors.append("Machine or cluster identifier in " + rel)
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=rel)
                python_files += 1
            except SyntaxError as exc:
                errors.append(f"Python syntax {rel}:{exc.lineno}")
        if path.suffix == ".json":
            try:
                json.loads(text)
            except ValueError:
                errors.append("Invalid JSON: " + rel)
        # Verbatim history has original workspace references; curated docs must have working local links.
        historical = rel.startswith("history/") or rel in {"PROJECT_RECORD_RSI_YSZ.md", "docs/experiments/environment-original.md"}
        if path.suffix == ".md" and not historical:
            for match in re.finditer(r"\[[^\]]*\]\(([^\s)]+)\)", text):
                target = match.group(1).strip("<>")
                if target.startswith(("http:", "https:", "mailto:", "#")):
                    continue
                target = unquote(target.split("#", 1)[0])
                if target and not (path.parent / target).exists():
                    errors.append("Broken curated link: " + rel + " -> " + target)
    manifest_path = ROOT / "provenance/manifest.json"
    if not manifest_path.exists():
        errors.append("Missing provenance manifest")
    else:
        for item in json.loads(manifest_path.read_text())["files"]:
            path = ROOT / item["published"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["published_sha256"]:
                errors.append("Published snapshot hash mismatch: " + item["published"])
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    result = {"status": "PASS", "public_files": len(files), "python_syntax_checked": python_files,
              "scope": "Public payload, credential patterns, curated links, JSON and snapshot hashes; not physics or training validation"}
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    check()
