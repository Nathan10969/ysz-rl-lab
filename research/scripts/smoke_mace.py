"""CUDA float64 E/F throughput and bounded fixed-cell FIRE relaxation smoke."""
import argparse
import json
import time
import traceback
from pathlib import Path
from smoke_common import (SMOKE, MODEL_SHA, Budget, distribution, map_final,
                          require_cuda_job, resources, save_json, sha256, versions)

LIMITS = {"wall_seconds": 1800, "raw_ef_calls": 1024, "batch_sizes": [1, 4, 16],
          "warmup_batches_per_size": 2, "steady_batches_per_size": 4,
          "relax_max_steps": 40, "relax_target_fmax_eV_A": 0.05, "relax_maxstep_A": 0.10}


def batch_indices(size, n_cells, offset=0):
    if type(size) is not int or type(n_cells) is not int or size < 1 or n_cells < 1:
        raise ValueError('Positive batch size and input count required')
    return [(offset + i) % n_cells for i in range(size)]


def relaxation_cases(n_cells, sizes=None):
    return [('batch_' + str(size), batch_indices(size, n_cells))
            for size in (LIMITS['batch_sizes'] if sizes is None else sizes)]


def build_batch(calc, group, device):
    """Construct actual AtomicData directly; its constructor cannot be called empty.

    Uses the installed MACECalculator's config/head/key convention. Do not
    de-batch its Batch: bundled PyG's reconstruction calls AtomicData() without
    its required fields before any model forward.
    """
    from mace import data
    from mace.tools import torch_geometric
    keys = data.KeySpecification(info_keys=calc.info_keys,
                                 arrays_keys=dict(calc.arrays_keys, **{calc.charges_key: 'charges'}))
    graphs = [data.AtomicData.from_config(
        data.config_from_atoms(cell, key_specification=keys, head_name=calc.head),
        z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)
        for cell in group]
    return torch_geometric.Batch.from_data_list(graphs).to(device)


def raw_ef_worst_case():
    sizes = LIMITS['batch_sizes']
    sp = sum(sizes) * (LIMITS['warmup_batches_per_size'] + LIMITS['steady_batches_per_size'])
    parity = max(sizes) + sum(size for size in sizes if size != 1)
    relax = sum(sizes) * (LIMITS['relax_max_steps'] + 1)
    return sp + parity + relax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=SMOKE / "outputs/mace")
    parser.add_argument("--preflight-only", action="store_true", help="Import/hash/data check; no model or GPU work")
    parser.add_argument("--cpu-execution-preflight", action="store_true",
                        help="Real CPU E/F and one FIRE step at batch1/4 through the job's code path")
    args = parser.parse_args()
    import numpy as np
    import torch
    from ase.io import read, write
    from ase.optimize import FIRE
    from mace.calculators import MACECalculator
    from mace.tools import torch_geometric
    actual_sha = sha256(args.checkpoint)
    if actual_sha != MODEL_SHA:
        raise ValueError("Frozen MP0b2 small checkpoint SHA256 mismatch")
    initial = read(SMOKE / "data/mace_initial_lattice.extxyz", index=":")
    cells = read(SMOKE / "data/mace_initial_physical.extxyz", index=":")
    assert len(initial) == len(cells) and raw_ef_worst_case() <= LIMITS['raw_ef_calls']
    for cell in cells:
        assert len(cell) == 316
        assert {z: int(np.sum(cell.numbers == z)) for z in [40, 39, 8]} == {40: 92, 39: 16, 8: 208}
    result = {"purpose": "frozen MLIP throughput smoke only; no DFT or RSI", "limits": LIMITS,
              "checkpoint_sha256": actual_sha, "dtype": "float64", "compute_stress": False,
              "versions": versions(["torch", "mace-torch", "ase", "spglib", "scipy", "e3nn"]),
              "data_sha256": sha256(SMOKE / "data/mace_initial_physical.extxyz"),
              "n_atoms": 316, "raw_ef_definition": "one energy+force evaluation for one 316-atom cell; batch B counts B",
              "input_unique_cells": len(cells), "worst_case_raw_ef_calls": raw_ef_worst_case(),
              "input_reuse_note": "Batch16 cycles the eight frozen inputs twice; each graph and relaxation copy is charged. This is throughput, not sixteen unique discoveries."}
    if args.preflight_only:
        # Maps ideal cells only; proves sublattice and canonical-key plumbing without MLIP work.
        result.update(status="imports_hash_data_preflight_passed", GPU_work_performed=False,
                      ideal_map=map_final(cells[0], initial[0]))
        save_json(args.output / "preflight.json", result)
        print(json.dumps({"status": result["status"], "checkpoint_sha256": actual_sha,
                          "versions": result["versions"]}))
        return
    device = 'cpu' if args.cpu_execution_preflight else 'cuda'
    if args.cpu_execution_preflight:
        torch.set_num_threads(2)
        result['runtime'] = {'device': 'cpu', 'GPU_work_performed': False, 'cpus': 2}
        result['cpu_preflight_scope'] = {'batch_sizes': [1, 4], 'SP_batches_per_size': 1,
                                       'FIRE_steps_per_copy': 1, 'production_code_path': True}
    else:
        result["runtime"] = require_cuda_job(torch)
    def synchronize():
        if device == 'cuda':
            torch.cuda.synchronize()
    sizes = [1, 4] if args.cpu_execution_preflight else LIMITS['batch_sizes']
    warmups = 0 if args.cpu_execution_preflight else LIMITS['warmup_batches_per_size']
    repeats = 1 if args.cpu_execution_preflight else LIMITS['steady_batches_per_size']
    max_steps = 1 if args.cpu_execution_preflight else LIMITS['relax_max_steps']
    torch.set_default_dtype(torch.float64)
    budget = Budget(LIMITS["wall_seconds"], LIMITS["raw_ef_calls"])
    args.output.mkdir(parents=True, exist_ok=True)
    calls = []
    t0 = time.monotonic()
    try:
        synchronize()
        start = time.monotonic()
        calc = MACECalculator(model_paths=str(args.checkpoint), device=device, default_dtype="float64",
                              enable_cueq=False, enable_oeq=False)
        assert len(calc.models) == 1 and next(calc.models[0].parameters()).dtype == torch.float64
        synchronize()
        result["load_seconds"] = time.monotonic() - start
        model = calc.models[0]

        def evaluate(group, phase):
            budget.check(len(group))
            synchronize()
            begin = time.monotonic()
            batch = build_batch(calc, group, device)
            synchronize()
            graph_seconds = time.monotonic() - begin
            forward_start = time.monotonic()
            # Count before invocation so failed forwards consume their intended raw-call budget.
            budget.charge(len(group))
            with (args.output / 'ef_attempts.jsonl').open('a') as handle:
                handle.write(json.dumps({'phase': phase, 'forward_call': budget.forward_calls,
                    'batch_size': len(group), 'raw_ef_calls_cumulative': budget.raw_ef_calls}) + '\n')
            output = model(batch.to_dict(), compute_stress=False, training=False,
                           compute_edge_forces=False, compute_atomic_stresses=False)
            energy = output["energy"].detach().cpu().numpy().copy()
            force = output["forces"].detach().cpu().numpy().copy()
            synchronize()
            forward_seconds = time.monotonic() - forward_start
            if not np.isfinite(energy).all() or not np.isfinite(force).all():
                raise RuntimeError("Non-finite MACE E/F output")
            forces = np.split(force, np.cumsum([len(cell) for cell in group])[:-1])
            row = {"phase": phase, "forward_call": budget.forward_calls, "batch_size": len(group),
                   "raw_ef_calls_cumulative": budget.raw_ef_calls,
                   "graph_transfer_seconds": graph_seconds, "forward_transfer_seconds": forward_seconds,
                   "seconds": time.monotonic() - begin, "energy_eV": energy.tolist(),
                   "fmax_eV_A": [float(np.linalg.norm(f, axis=1).max()) for f in forces],
                   **resources(torch)}
            calls.append(row)
            with (args.output / "ef_calls.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps({key: row[key] for key in ["phase", "forward_call", "batch_size", "seconds", "raw_ef_calls_cumulative"]}), flush=True)
            del output, batch
            return energy, forces

        # Each batch size has its own cold/warmup measurements; never cached ASE results.
        throughput = {}
        for size in sizes:
            group = [cells[i] for i in batch_indices(size, len(cells))]
            warmup = []
            for _ in range(warmups):
                evaluate(group, f"warmup_b{size}")
                warmup.append(calls[-1]["seconds"])
            times = []
            for repeat in range(repeats):
                group = [cells[i] for i in batch_indices(size, len(cells), repeat * size)]
                evaluate(group, f"steady_b{size}")
                times.append(calls[-1]["seconds"])
            throughput[str(size)] = {"warmup_seconds": warmup, "steady": distribution(times),
                                     "cells_per_second": size * len(times) / sum(times)}
        result["single_batch_SP"] = throughput
        parity_ids = batch_indices(max(sizes), len(cells))
        individual = [evaluate([cells[i]], "batch_equivalence_single") for i in parity_ids]
        parity = {}
        parity_sizes = [size for size in sizes if size != 1]
        result["batch_equivalence"] = {"batch_sizes_checked": parity_sizes, "per_batch_size": parity,
                                       "energy_tolerance_eV": 1e-6, "force_tolerance_eV_A": 1e-6}
        for size in parity_sizes:
            batched_e, batched_f = evaluate([cells[i] for i in parity_ids[:size]], f"batch_equivalence_b{size}")
            max_de = float(max(abs(individual[i][0][0] - batched_e[i]) for i in range(size)))
            max_df = float(max(np.abs(individual[i][1][0] - batched_f[i]).max() for i in range(size)))
            parity[str(size)] = {"max_energy_difference_eV": max_de, "max_force_difference_eV_A": max_df,
                                 "passed": max_de <= 1e-6 and max_df <= 1e-6}
            if not parity[str(size)]["passed"]:
                raise RuntimeError(f"Singleton/batch{size} E/F parity check failed")
        result["batch_equivalence"].update(
            max_energy_difference_eV=max(row["max_energy_difference_eV"] for row in parity.values()),
            max_force_difference_eV_A=max(row["max_force_difference_eV_A"] for row in parity.values()))

        relax_results = []
        for name, ids in relaxation_cases(len(cells), sizes):
            group = [cells[i].copy() for i in ids]
            optimizers = [FIRE(cell, logfile=None, dt=0.05, maxstep=LIMITS["relax_maxstep_A"],
                               downhill_check=False) for cell in group]
            active = list(range(len(group)))
            last = {}
            nsteps = [0] * len(group)
            step_rows = []
            relax_start = time.monotonic()
            for iteration in range(max_steps + 1):
                iteration_start = time.monotonic()
                budget.check()
                if not active:
                    break
                energy, forces = evaluate([group[i] for i in active], f"relax_{name}")
                next_active = []
                for local, i in enumerate(active):
                    fmax = float(np.linalg.norm(forces[local], axis=1).max())
                    last[i] = {"energy_eV": float(energy[local]), "fmax_eV_A": fmax,
                               "converged": fmax <= LIMITS["relax_target_fmax_eV_A"]}
                    if not last[i]["converged"] and iteration < max_steps:
                        # FIRE's supplied-force API takes no hidden extra model evaluation.
                        optimizers[i].step(forces[local])
                        optimizers[i].nsteps += 1
                        nsteps[i] += 1
                        next_active.append(i)
                step_rows.append({"iteration": iteration, "active_cells": len(active),
                                  "seconds": time.monotonic() - iteration_start,
                                  "ef_seconds": calls[-1]["seconds"], "resources": resources(torch)})
                active = next_active
            group_wall_seconds = time.monotonic() - relax_start
            for i, cell in enumerate(group):
                mapped = map_final(cell, initial[ids[i]])
                write(args.output / f"relax_{name}_{i}.extxyz", cell)
                relax_results.append({"case": name, "input_cell_id": ids[i], "steps": nsteps[i],
                                      "score": last[i]["energy_eV"], "score_status": "converged" if last[i]["converged"] else "step_capped_unconverged",
                                      **last[i], "final_occupation": mapped,
                                      "group_wall_seconds": group_wall_seconds,
                                      "step_timing": distribution([row["seconds"] for row in step_rows]),
                                      "step_records": step_rows})
            save_json(args.output / "relaxations.json", relax_results)
        result.update(status="cpu_execution_preflight_passed" if args.cpu_execution_preflight else "completed",
                      relaxations=relax_results)
    except Exception as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(raw_ef_calls=budget.raw_ef_calls, energy_calls=budget.raw_ef_calls,
                      force_calls=budget.raw_ef_calls, model_forward_calls=budget.forward_calls,
                      wall_seconds=time.monotonic() - t0, resources=resources(torch))
        save_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
