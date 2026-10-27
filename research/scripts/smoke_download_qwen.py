"""Download public official Qwen model files only; never loads weights or uses tokens."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote
import requests

ALLOWED = {"LICENSE", "README.md", "config.json", "configuration.json", "generation_config.json",
           "merges.txt", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
API = "https://modelscope.cn/api/v1/models/Qwen/Qwen3-0.6B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    target = args.destination
    target.mkdir(parents=True, exist_ok=True)
    response = requests.get(API + "/repo/files", params={"Revision": "master", "Recursive": "true"}, timeout=30)
    response.raise_for_status()
    metadata = response.json()
    files = [row for row in metadata["Data"]["Files"] if row["Path"] in ALLOWED]
    if {row["Path"] for row in files} != ALLOWED or sum(row["Size"] for row in files) > 1_600_000_000:
        raise ValueError("Unexpected official model file list/size")
    manifest = {"repository": "Qwen/Qwen3-0.6B", "official_modelscope": "https://modelscope.cn/models/Qwen/Qwen3-0.6B",
                "HF_origin": "https://huggingface.co/Qwen/Qwen3-0.6B", "public_download_no_token": True,
                "retrieved_unix": time.time(), "files": [], "status": "downloading"}
    (target.parent / "qwen_source_listing.json").write_text(json.dumps(metadata, indent=2))
    for row in files:
        path = target / row["Path"]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == row["Sha256"]:
            digest = row["Sha256"]
        else:
            # File commits and SHA256 from the official repository listing pin each byte stream.
            url = API + "/repo?Revision=" + quote(row["Revision"]) + "&FilePath=" + quote(row["Path"])
            download = requests.get(url, stream=True, timeout=(30, 90))
            download.raise_for_status()
            partial = path.with_suffix(path.suffix + ".partial")
            hashed = hashlib.sha256()
            size = 0
            with partial.open("wb") as handle:
                for chunk in download.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > row["Size"]:
                        raise ValueError("Download exceeds official file size")
                    handle.write(chunk)
                    hashed.update(chunk)
            digest = hashed.hexdigest()
            if size != row["Size"] or digest != row["Sha256"]:
                raise ValueError("Official file size/SHA256 check failed: " + row["Path"])
            partial.replace(path)
        manifest["files"].append({"path": row["Path"], "bytes": row["Size"], "sha256": digest,
                                  "revision": row["Revision"]})
        (target.parent / "qwen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("verified", row["Path"], row["Size"], digest, flush=True)
    config = json.loads((target / "config.json").read_text())
    assert config["model_type"] == "qwen3" and config["hidden_size"] == 1024
    manifest["status"] = "verified_complete"
    (target.parent / "qwen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Official Qwen3-0.6B download complete; no model load or GPU work.", flush=True)


if __name__ == "__main__":
    main()
