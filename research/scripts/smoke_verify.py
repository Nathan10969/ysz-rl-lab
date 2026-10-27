"""CPU-only checks of smoke data, physical invariants, manifests, and hard caps."""
import json
import re
from collections import Counter
import numpy as np
from ase import Atoms
from ase.io import read
from lattice import apply_action
from smoke_common import SMOKE, ROOT, map_final, save_json, sha256


def main():
    manifest = json.loads((SMOKE / "JOB_MANIFEST.json").read_text())
    budget = json.loads((SMOKE / "BUDGET.json").read_text())
    data = json.loads((SMOKE / "data/manifest.json").read_text())
    assert budget["authorized_card_hours"] == 10
    assert budget["independent_of_DFT_40_gpu_hour_pool"]
    assert not budget["transfers_between_pools_permitted"]
    assert sum(row["hard_gpu_hours"] for row in manifest["jobs"]) == budget["planned_hard_gpu_hours"]
    assert budget["planned_hard_gpu_hours"] <= 10
    for row in manifest["jobs"]:
        sub = (SMOKE / row["sub"]).read_text()
        hours, minutes, seconds = map(int, re.search(r"#SBATCH --time=(\d+):(\d+):(\d+)", sub).groups())
        assert row["gpus"] == 1 and "#SBATCH --gres=gpu:a100:1" in sub
        assert row["gpus"] * (hours + minutes / 60 + seconds / 3600) == row["hard_gpu_hours"] <= 2
        assert "--no-requeue" in sub
    checked = 0
    for name, expected_n in [("train.jsonl", 64), ("validation.jsonl", 4)]:
        rows = [json.loads(line) for line in (SMOKE / "data" / name).read_text().splitlines()]
        assert len(rows) == expected_n
        for row in rows:
            assert not any("energy" in key.lower() or "DFT" in key for key in row)
            before = Atoms(numbers=row["numbers"])
            after = apply_action(before, row["action"])
            assert Counter(before.numbers) == Counter(after.numbers) == {40: 92, 39: 16, 8: 208, 0: 8}
            checked += 1
    for name, digest in data["files"].items():
        assert sha256(SMOKE / "data" / name) == digest
    physical = read(SMOKE / "data/mace_initial_physical.extxyz", index=":")
    lattice = read(SMOKE / "data/mace_initial_lattice.extxyz", index=":")
    assert len(physical) == len(lattice) == 8
    assert all(len(cell) == 316 and np.all(cell.numbers != 0) for cell in physical)
    mapping = map_final(physical[0], lattice[0])
    assert mapping["canonical_key"] == mapping["initial_canonical_key"]
    assert mapping["max_assignment_distance_A"] < 1e-7
    result = {"status": "passed", "legal_action_examples": checked,
              "physical_cells": len(physical), "GPU_work_performed": False,
              "MLIP_calls": 0, "hard_reserved_card_hours": budget["planned_hard_gpu_hours"],
              "ideal_canonical_key": mapping["canonical_key"],
              "scope": "static, physical-invariant, data-integrity and budget checks only; no model computation"}
    save_json(SMOKE / "evidence/static_verification.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "ideal_canonical_key"}))


if __name__ == "__main__":
    main()
