"""Generate unscored legal-action data. Does not run an MLIP or load an LLM."""
import json
from collections import Counter
import numpy as np
from ase.io import write
from calibration import make_state
from lattice import apply_action, physical_atoms
from smoke_common import SMOKE, save_json, sha256

SYSTEM = ('You control a fixed fluorite lattice with 324 numbered sites. '
          'Return exactly one JSON action and no other text. '
          'swap_cations exchanges one Y and one Zr. '
          'move_vacancy exchanges one vacant oxygen site and one occupied oxygen site. '
          'Preserve composition. Follow the requested operation. /no_think')


def main():
    data = SMOKE / "data"
    data.mkdir(parents=True, exist_ok=True)
    rows = []
    states = []
    for index in range(68):
        seed = 2026102300 + index
        state = make_state(seed, 3, 16)
        rng = np.random.default_rng(seed + 1000)
        op = "swap_cations" if index % 2 == 0 else "move_vacancy"
        left, right = (39, 40) if op == "swap_cations" else (0, 8)
        action = {"op": op, "sites": [int(rng.choice(np.flatnonzero(state.numbers == left))),
                                      int(rng.choice(np.flatnonzero(state.numbers == right)))]}
        after = apply_action(state, action)
        assert Counter(state.numbers) == Counter(after.numbers)
        prompt = {"template": "fluorite_3x3x3_a5.14_A", "n_sites": 324,
                  "Y_sites": np.flatnonzero(state.numbers == 39).tolist(),
                  "vacancy_sites": np.flatnonzero(state.numbers == 0).tolist(),
                  "requested_op": op,
                  "candidate_sites": action["sites"],
                  "task": "Emit the legal action using the two candidate sites."}
        # Candidate sites deliberately make this a format/legality smoke, not an energy policy.
        rows.append({"id": f"smoke_{index:03d}", "seed": seed,
                     "messages": [{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))}],
                     "action": action, "numbers": state.numbers.tolist(),
                     "purpose": "unscored synthetic legal action; workflow smoke only"})
        if index < 8:
            states.append(state)
    for name, subset in [("train.jsonl", rows[:64]), ("validation.jsonl", rows[64:])]:
        (data / name).write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in subset))
    write(data / "mace_initial_lattice.extxyz", states)
    write(data / "mace_initial_physical.extxyz", [physical_atoms(state) for state in states])
    save_json(data / "manifest.json", {
        "purpose": "workflow and throughput smoke only; not high-quality demonstrations",
        "DFT_labels_used": False, "MLIP_labels_used": False, "MLIP_calls_for_generation": 0,
        "train_examples": 64, "validation_examples": 4, "mace_geometries": 8,
        "composition": {"Zr": 92, "Y": 16, "O": 208}, "n_lattice_sites": 324,
        "n_physical_atoms": 316, "seed_base": 2026102300,
        "files": {name: sha256(data / name) for name in
                  ["train.jsonl", "validation.jsonl", "mace_initial_lattice.extxyz", "mace_initial_physical.extxyz"]}})
    print("Prepared 64 train, 4 validation legal actions and 8 unscored 316-atom cells.")


if __name__ == "__main__":
    main()
