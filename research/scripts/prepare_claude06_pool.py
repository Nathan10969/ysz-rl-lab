"""Prepare a bounded local-only 316-atom YSZ pool; never submit or train.

The low-energy group is selected by UNRELAXED frozen-MACE energy.  No DFT
result, label, ranking, or file enters generation, selection, or deduplication.
"""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import find_mic
from ase.io import read, write
from ase.optimize import BFGS
from scipy.optimize import linear_sum_assignment

from calibration import canonical_key, make_state, reference, shell_counts
from prepare_calibration import make_calc, physical


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "calibration/claude06_pool"
MODEL = ROOT / "models/mace-small-density-agnesi-stress.model"
SEED_BASE = 20261060
POOL_SIZE = 64
TARGET_SIZE = 16
FMAX = 0.03
EXPECTED = {"Zr": 92, "Y": 16, "O": 208}


def save_json(name, data):
    """Keep checkpoints readable if preparation is interrupted."""
    path = OUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def assert_composition(atoms):
    assert dict(Counter(atoms.get_chemical_symbols())) == EXPECTED
    assert len(atoms) == 316
    assert 4 * EXPECTED["Zr"] + 3 * EXPECTED["Y"] - 2 * EXPECTED["O"] == 0


def map_relaxed(atoms, initial_state):
    """Audit-style periodic Hungarian assignment on each parent sublattice.

    Zr and Y share the cation sublattice, so they must compete in the SAME
    assignment; independent element assignments could reuse a cation site.
    Oxygen competes on the oxygen sublattice, leaving eight vacancy sites.
    """
    parent = reference(3)
    assigned = np.full(len(atoms), -1, dtype=int)
    distances = np.full(len(atoms), np.nan)
    for atom_mask, sites in [
        (np.isin(atoms.numbers, [39, 40]), np.flatnonzero(parent.numbers == 40)),
        (atoms.numbers == 8, np.flatnonzero(parent.numbers == 8)),
    ]:
        ai = np.flatnonzero(atom_mask)
        vectors = atoms.positions[ai, None, :] - parent.positions[sites][None, :, :]
        cost = find_mic(vectors.reshape(-1, 3), parent.cell, pbc=True)[1]
        cost = cost.reshape(len(ai), len(sites))
        ii, jj = linear_sum_assignment(cost ** 2)
        assigned[ai[ii]] = sites[jj]
        distances[ai[ii]] = cost[ii, jj]
    assert np.all(assigned >= 0)
    assert len(set(assigned.tolist())) == len(atoms)
    assert np.all(parent.numbers[assigned[atoms.numbers == 8]] == 8)
    assert np.all(parent.numbers[assigned[np.isin(atoms.numbers, [39, 40])]] == 40)
    mapped = parent.copy()
    mapped.numbers[:] = 0
    mapped.numbers[assigned] = atoms.numbers
    original_ids = atoms.arrays["site_id"]
    displacement = find_mic(
        atoms.positions - parent.positions[original_ids], parent.cell, pbc=True
    )[1]
    initial_key = canonical_key(initial_state, 3)
    final_key = canonical_key(mapped, 3)
    return mapped, assigned, {
        "canonical_key_relaxed_site_mapping": final_key,
        "occupation_changed": bool(np.any(mapped.numbers != initial_state.numbers)),
        "occupation_class_changed": initial_key != final_key,
        "n_atoms_mapped_to_different_sites": int(np.sum(assigned != original_ids)),
        "max_displacement_from_initial_site_A": float(displacement.max()),
        "max_final_site_assignment_distance_A": float(distances.max()),
        "assignment_distances_A": distances.tolist(),
        "initial_atom_site_ids": original_ids.tolist(),
        "relaxed_atom_site_ids": assigned.tolist(),
        "initial_vacancy_site_ids": np.flatnonzero(initial_state.numbers == 0).tolist(),
        "relaxed_vacancy_site_ids": np.flatnonzero(mapped.numbers == 0).tolist(),
        "NN_NNN_before": shell_counts(initial_state),
        "NN_NNN_after_mapping": shell_counts(mapped),
    }


def selection_plan(candidates):
    random_indices = list(range(6))
    low_ranking = sorted(range(6, POOL_SIZE), key=lambda i: (candidates[i]["ideal_MACE_energy_eV"], i))
    low_indices = low_ranking[:6]
    remaining = [i for i in range(POOL_SIZE) if i not in random_indices + low_indices]
    nn_low = sorted(remaining, key=lambda i: (candidates[i]["Y_vac_NN"], i))
    nn_low_indices = nn_low[:2]
    nn_high = sorted(
        [i for i in remaining if i not in nn_low_indices],
        key=lambda i: (-candidates[i]["Y_vac_NN"], i),
    )
    plan = [("random", i, list(range(POOL_SIZE))) for i in random_indices]
    plan += [("unrelaxed_MACE_low", i, low_ranking) for i in low_indices]
    plan += [("NN_boundary_low", i, nn_low) for i in nn_low_indices]
    plan += [("NN_boundary_high", i, nn_high) for i in nn_high[:2]]
    assert len(plan) == TARGET_SIZE
    assert len({i for _, i, _ in plan}) == TARGET_SIZE
    return plan


def verify_saved(require_complete=True):
    manifest = json.loads((OUT / "candidates.json").read_text())
    records = json.loads((OUT / "configurations.json").read_text())
    candidates = manifest["candidates"]
    assert len(candidates) == POOL_SIZE
    assert [r["seed"] for r in candidates] == list(range(SEED_BASE, SEED_BASE + POOL_SIZE))
    assert len({r["canonical_key_initial"] for r in candidates}) == POOL_SIZE
    assert all(np.isfinite(r["ideal_MACE_energy_eV"]) for r in candidates)
    assert len({r["canonical_key_initial"] for r in records}) == len(records)
    assert len({r["canonical_key_relaxed_site_mapping"] for r in records}) == len(records)
    assert len({r["pool_index"] for r in records}) == len(records)
    assert [r["id"] for r in records] == [f"B{i:02d}" for i in range(len(records))]
    plan = selection_plan(candidates)
    expected_plan = [{"id": f"B{i:02d}", "selection_group": group, "pool_index": index}
                     for i, (group, index, _) in enumerate(plan)]
    assert manifest["planned_selection"] == expected_plan
    protected = {index for _, index, _ in plan}
    attempted = set()
    accepted_attempts = []
    expected_slot = 0
    for attempt in manifest["relaxation_attempts"]:
        slot = int(attempt["id"][1:])
        assert slot == expected_slot
        group, first, fallback = plan[slot]
        choices = [first] + [i for i in fallback if i not in protected]
        next_index = next(i for i in choices if i not in attempted)
        assert attempt["pool_index"] == next_index
        assert attempt["selection_group"] == group
        assert attempt["seed"] == SEED_BASE + next_index
        attempted.add(next_index)
        if attempt["accepted"]:
            assert attempt["converged"] and not attempt["duplicate_relaxed_class"]
            assert attempt["max_force_eV_A"] <= FMAX
            accepted_attempts.append(attempt)
            expected_slot += 1
    assert len(accepted_attempts) == len(records)
    for attempt, rec in zip(accepted_attempts, records):
        assert all(rec[key] == value for key, value in attempt.items())
    assert manifest["selected"] == [
        {key: rec[key] for key in ["id", "pool_index", "seed", "selection_group"]}
        for rec in records
    ]
    if require_complete:
        assert len(records) == TARGET_SIZE, f"Only {len(records)}/16 unique converged structures"
        assert Counter(r["selection_group"] for r in records) == {
            "random": 6, "unrelaxed_MACE_low": 6,
            "NN_boundary_low": 2, "NN_boundary_high": 2,
        }
    maxima = []
    for slot, rec in enumerate(records):
        group, first, fallback = plan[slot]
        assert rec["selection_group"] == group
        assert rec["original_planned_pool_index"] == first
        assert rec["seed"] == SEED_BASE + rec["pool_index"]
        assert rec["replaced_original_candidate"] == (rec["pool_index"] != first)
        assert rec["pool_index"] in [first] + [i for i in fallback if i not in protected]
        state = make_state(rec["seed"], 3, 16)
        assert canonical_key(state, 3) == rec["canonical_key_initial"]
        initial = read(OUT / f'{rec["id"]}_initial.extxyz')
        relaxed = read(OUT / f'{rec["id"]}_relaxed.extxyz')
        assert_composition(initial)
        assert_composition(relaxed)
        expected_initial = physical(state)
        assert np.array_equal(initial.numbers, expected_initial.numbers)
        assert np.array_equal(initial.arrays["site_id"], expected_initial.arrays["site_id"])
        assert np.array_equal(initial.arrays["site_id"], relaxed.arrays["site_id"])
        assert np.array_equal(initial.numbers, relaxed.numbers)
        assert np.allclose(initial.positions, physical(state).positions)
        assert np.isclose(initial.get_potential_energy(), rec["initial_energy_eV"], rtol=0, atol=1e-7)
        assert np.isclose(initial.get_potential_energy(), candidates[rec["pool_index"]]["ideal_MACE_energy_eV"], rtol=0, atol=1e-7)
        for atoms in [initial, relaxed]:
            assert atoms.get_forces().shape == (316, 3)
            assert np.all(np.isfinite(atoms.get_forces()))
        max_force = float(np.linalg.norm(relaxed.get_forces(), axis=1).max())
        assert rec["converged"] and max_force <= FMAX + 1e-8
        assert np.isclose(max_force, rec["max_force_eV_A"], rtol=0, atol=1e-8)
        assert np.isclose(relaxed.get_potential_energy(), rec["relaxed_energy_eV"], rtol=0, atol=1e-7)
        _, assigned, audit = map_relaxed(relaxed, state)
        assert assigned.tolist() == rec["relaxed_atom_site_ids"]
        assert np.array_equal(assigned, relaxed.arrays["assigned_site_id"])
        assert audit["canonical_key_relaxed_site_mapping"] == rec["canonical_key_relaxed_site_mapping"]
        assert audit["NN_NNN_after_mapping"] == rec["NN_NNN_after_mapping"]
        assert rec["model_sha256"] == hashlib.sha256(MODEL.read_bytes()).hexdigest()
        maxima.append(max_force)
    result = {
        "verified_records": len(records), "candidate_count": len(candidates),
        "unique_initial_classes": len({r["canonical_key_initial"] for r in records}),
        "unique_relaxed_mapped_classes": len({r["canonical_key_relaxed_site_mapping"] for r in records}),
        "composition": EXPECTED, "charge_e": 0,
        "all_saved_force_arrays_finite": True,
        "largest_max_force_eV_A": max(maxima) if maxima else None,
        "force_threshold_eV_A": FMAX, "complete": len(records) == TARGET_SIZE,
        "DFT_used": False, "training_performed": False, "remote_actions": 0,
    }
    save_json("verification.json", result)
    print("VERIFICATION " + json.dumps(result), flush=True)
    return result


def generate_pool(calc, model_hash):
    states, candidates = [], []
    for i in range(POOL_SIZE):
        state = make_state(SEED_BASE + i, 3, 16)
        atoms = physical(state)
        assert_composition(atoms)
        atoms.calc = calc
        calc.reset()
        energy = float(atoms.get_potential_energy())
        candidate = {
            "pool_index": i, "seed": SEED_BASE + i,
            "canonical_key_initial": canonical_key(state, 3),
            "ideal_MACE_energy_eV": energy,
            "ideal_MACE_energy_per_atom_eV": energy / len(atoms),
            "ideal_vacancy_site_ids": np.flatnonzero(state.numbers == 0).tolist(),
            "ideal_Y_site_ids": np.flatnonzero(state.numbers == 39).tolist(),
            **shell_counts(state),
        }
        states.append(state)
        candidates.append(candidate)
        print(f"IDEAL {i:02d} seed={SEED_BASE+i} energy_eV={energy:.10f} NN={candidate['Y_vac_NN']}", flush=True)
    assert len({r["canonical_key_initial"] for r in candidates}) == POOL_SIZE
    plan = selection_plan(candidates)
    manifest = {
        "schema_version": 1, "purpose": "Claude06 local frozen-MACE maincell candidate pool",
        "n": 3, "n_sites": 324, "n_atoms": 316, "composition": EXPECTED,
        "charge_e": 0, "n_vacancies": 8, "seed_base": SEED_BASE,
        "candidate_count": POOL_SIZE, "target_count": TARGET_SIZE,
        "generation": "calibration.make_state(seed_base+i, n=3, n_y=16)",
        "selection_basis": "6 first-seed random; 6 lowest UNRELAXED MACE energy excluding first six; 2 lowest and 2 highest Y-vac NN from remaining (stable pool-index ties)",
        "low_group_caveat": "Selected by UNRELAXED frozen-MACE single-point energy; not DFT energy and not relaxed-energy ranking",
        "replacement_policy": "On failed convergence or relaxed mapped symmetry equivalence, take next unused/unreserved candidate in the same group's ranking; stop and report if pool or local time budget exhausted",
        "initial_dedupe": "Full fluorite supercell spglib symmetry canonical_key",
        "relaxed_dedupe": "Periodic one-to-one Hungarian cation/oxygen sublattice assignment, preserving Zr/Y species, then full canonical symmetry comparison",
        "model": str(MODEL.relative_to(ROOT)), "model_sha256": model_hash,
        "preparation_device": "cpu", "preparation_dtype": "float64", "torch_num_threads": 4,
        "relaxation": {"optimizer": "ASE BFGS", "fmax_eV_A": FMAX, "max_steps": 300, "maxstep_A": 0.1, "fixed_cell": True},
        "DFT_used": False, "training_performed": False, "remote_actions": 0,
        "candidates": candidates,
        "planned_selection": [{"id": f"B{i:02d}", "selection_group": group, "pool_index": index} for i, (group, index, _) in enumerate(plan)],
        "relaxation_attempts": [], "selected": [], "issues": [],
    }
    return states, manifest


def finish_pool(states, manifest, calc, started, max_seconds):
    model_hash = manifest["model_sha256"]
    candidates = manifest["candidates"]
    plan = selection_plan(candidates)
    records = json.loads((OUT / "configurations.json").read_text())
    attempted = {r["pool_index"] for r in manifest["relaxation_attempts"]}
    relaxed_keys = {r["canonical_key_relaxed_site_mapping"] for r in records}
    prior_seconds = manifest.get("local_preparation_seconds", 0.0)
    manifest["local_preparation_budget_seconds"] = max_seconds
    protected = {index for _, index, _ in plan}
    for slot, (group, first, fallback) in enumerate(plan):
        if slot < len(records):
            continue
        cid = f"B{slot:02d}"
        choices = [first] + [i for i in fallback if i not in protected]
        accepted = False
        for index in choices:
            if index in attempted:
                continue
            if prior_seconds + time.perf_counter() - started >= max_seconds:
                manifest["issues"].append(f"Local time budget exhausted before {cid}")
                break
            attempted.add(index)
            candidate = candidates[index]
            state = states[index]
            atoms = physical(state)
            atoms.calc = calc
            calc.reset()
            initial_energy = float(atoms.get_potential_energy())
            initial_forces = atoms.get_forces().copy()
            assert np.isclose(initial_energy, candidate["ideal_MACE_energy_eV"], rtol=0, atol=1e-7)
            initial = atoms.copy()
            initial.calc = SinglePointCalculator(initial, energy=initial_energy, forces=initial_forces)
            t0 = time.perf_counter()
            attempt_number = len(manifest["relaxation_attempts"])
            dyn = BFGS(atoms, logfile=str(OUT / f"{cid}_attempt{attempt_number:02d}.log"), maxstep=0.1)
            converged = bool(dyn.run(fmax=FMAX, steps=300))
            forces = atoms.get_forces().copy()
            energy = float(atoms.get_potential_energy())
            max_force = float(np.linalg.norm(forces, axis=1).max())
            mapped, assigned, audit = map_relaxed(atoms, state)
            duplicate = audit["canonical_key_relaxed_site_mapping"] in relaxed_keys
            success = converged and max_force <= FMAX and not duplicate
            attempt = {
                "id": cid, "selection_group": group, "pool_index": index,
                "seed": candidate["seed"], "converged": converged, "steps": dyn.nsteps,
                "max_force_eV_A": max_force, "relax_seconds": time.perf_counter() - t0,
                "duplicate_relaxed_class": duplicate, "accepted": success,
            }
            manifest["relaxation_attempts"].append(attempt)
            manifest["local_preparation_seconds"] = prior_seconds + time.perf_counter() - started
            print("RELAX " + json.dumps(attempt), flush=True)
            if not success:
                save_json("candidates.json", manifest)
                continue
            relaxed_keys.add(audit["canonical_key_relaxed_site_mapping"])
            rec = {
                **attempt, "n_atoms": len(atoms), "n_sites": len(state), "n_y": 16,
                "canonical_key_initial": candidate["canonical_key_initial"],
                "canonical_key": candidate["canonical_key_initial"],
                "initial_energy_eV": initial_energy, "relaxed_energy_eV": energy,
                "pool_unrelaxed_MACE_energy_eV": candidate["ideal_MACE_energy_eV"],
                "energy_selection_label": "UNRELAXED MACE energy" if group == "unrelaxed_MACE_low" else "no energy ranking for this group",
                "preparation_device": "cpu", "preparation_dtype": "float64", "torch_num_threads": 4,
                "model_sha256": model_hash, "composition": EXPECTED, "charge_e": 0,
                "initial_max_force_eV_A": float(np.linalg.norm(initial_forces, axis=1).max()),
                "replaced_original_candidate": index != first,
                "original_planned_pool_index": first,
                **shell_counts(state), **audit,
            }
            atoms.set_array("assigned_site_id", assigned)
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
            for geometry, kind in [(initial, "initial"), (atoms, "relaxed")]:
                geometry.info.update(id=cid, seed=candidate["seed"], selection_group=group,
                                     model_sha256=model_hash, geometry_kind=kind)
                write(OUT / f"{cid}_{kind}.extxyz", geometry, format="extxyz")
            records.append(rec)
            manifest["selected"].append({"id": cid, "pool_index": index, "seed": candidate["seed"], "selection_group": group})
            save_json("configurations.json", records)
            save_json("candidates.json", manifest)
            accepted = True
            break
        if not accepted:
            manifest["issues"].append(f"Could not fill {cid} from same-group unused candidates within local budget")
            break
    manifest["local_preparation_seconds"] = prior_seconds + time.perf_counter() - started
    manifest["complete"] = len(records) == TARGET_SIZE
    save_json("candidates.json", manifest)
    verify_saved(require_complete=manifest["complete"])
    print("PREPARATION " + json.dumps({"complete": manifest["complete"], "selected": len(records), "seconds": manifest["local_preparation_seconds"], "issues": manifest["issues"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify saved files without rerunning MACE")
    parser.add_argument("--resume", action="store_true", help="Continue verified checkpoints without repeating the 64 ideal SPs")
    parser.add_argument("--max-seconds", type=float, default=1800.0,
                        help="Total preparation budget including earlier checkpointed runs")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.verify:
        verify_saved()
        return
    existing = (OUT / "configurations.json").exists()
    if existing and not args.resume:
        raise RuntimeError("Existing preparation found; use --verify or --resume")
    if args.resume and not existing:
        raise RuntimeError("No checkpoint exists to resume")
    started = time.perf_counter()
    torch.set_num_threads(4)
    model_hash = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if args.resume:
        manifest = json.loads((OUT / "candidates.json").read_text())
        assert manifest["model_sha256"] == model_hash
        verify_saved(require_complete=False)
        states = [make_state(SEED_BASE + i, 3, 16) for i in range(POOL_SIZE)]
        assert [canonical_key(s, 3) for s in states] == [r["canonical_key_initial"] for r in manifest["candidates"]]
        manifest.setdefault("previous_run_issues", []).extend(manifest["issues"])
        manifest["issues"] = []
        manifest["resumed"] = True
        print(f"RESUME accepted={len(manifest['selected'])} elapsed_seconds={manifest.get('local_preparation_seconds', 0):.3f}", flush=True)
        calc = make_calc("cpu", "float64")
    else:
        calc = make_calc("cpu", "float64")
        states, manifest = generate_pool(calc, model_hash)
        save_json("candidates.json", manifest)
        save_json("configurations.json", [])
    finish_pool(states, manifest, calc, started, args.max_seconds)


if __name__ == "__main__":
    main()
