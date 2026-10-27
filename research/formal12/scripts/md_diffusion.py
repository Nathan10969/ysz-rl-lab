"""Review12 MD: oxygen diffusion with the frozen MACE model (NVT Langevin, fixed cell).

One process = one structure at one temperature. Several processes share one
A100 (MACE small on 316 atoms leaves the GPU mostly idle). Checkpoints allow a
run to continue across Slurm jobs. Cations do not hop on ns time scales, so the
comparison measures how the frozen Y/Zr arrangement (from each method's
low-energy terminals) affects O transport.

Outputs: o_positions.npy (unwrapped O, every --sample steps), thermo.csv,
checkpoint.npz, summary.json (MSD slope -> D_O in cm^2/s).
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_structure(path, key=None):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and 'terminals' in data:
        rows = {r['terminal_key']: r for r in data['terminals']}
        data = rows[key]
    from ase import Atoms
    return Atoms(numbers=data['numbers'], positions=data['positions_A'], cell=data['cell_A'], pbc=True)


def diffusion_from_msd(times_ps, msd_A2, fit_from=0.5):
    """D = slope/6 over the later part of the MSD curve; returns cm^2/s."""
    n = len(times_ps)
    start = int(n * fit_from)
    slope, intercept = np.polyfit(times_ps[start:], msd_A2[start:], 1)
    return slope / 6 * 1e-4, slope, intercept   # A^2/ps -> cm^2/s factor 1e-16/1e-12


def msd(positions):
    """Mean over O of |r(t)-r(0)|^2 for each sampled frame (Angstrom^2)."""
    displacement = positions - positions[0]
    return (displacement ** 2).sum(-1).mean(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structure', type=Path, required=True, help='JSON with numbers/positions_A/cell_A or a result.json')
    parser.add_argument('--key', default=None)
    parser.add_argument('--checkpoint-model', type=Path, required=True)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--steps', type=int, required=True, help='production steps for this invocation')
    parser.add_argument('--equilibration', type=int, default=10000)
    parser.add_argument('--dt-fs', type=float, default=2.0)
    parser.add_argument('--friction', type=float, default=0.01, help='1/fs')
    parser.add_argument('--sample', type=int, default=250)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dtype', default='float32')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--wall-seconds', type=float, default=6000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    begin = time.monotonic()
    import torch
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from mace.calculators import MACECalculator
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    atoms = load_structure(args.structure, args.key)
    atoms.calc = MACECalculator(model_paths=str(args.checkpoint_model), device=args.device, default_dtype=args.dtype)
    oxygen = np.flatnonzero(atoms.numbers == 8)
    ckpt = out / 'checkpoint.npz'
    rng = np.random.default_rng(args.seed)
    if ckpt.exists():
        state = np.load(ckpt)
        atoms.set_positions(state['positions']); atoms.set_momenta(state['momenta'])
        done, samples = int(state['done']), list(np.load(out / 'o_positions.npy'))
        equilibrated = True
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)
        done, samples, equilibrated = 0, [], False
    dyn = Langevin(atoms, args.dt_fs * units.fs, temperature_K=args.temperature, friction=args.friction / units.fs,
                   rng=np.random.default_rng(args.seed + done + 1))
    thermo = (out / 'thermo.csv').open('a')
    if not equilibrated:
        t0 = time.monotonic()
        dyn.run(args.equilibration)
        thermo.write(f'equilibration,{args.equilibration},{atoms.get_temperature():.1f},{time.monotonic() - t0:.1f}\n')
        samples = [atoms.get_positions()[oxygen].copy()]
    t0, target = time.monotonic(), done + args.steps
    while done < target and time.monotonic() - begin < args.wall_seconds:
        dyn.run(args.sample); done += args.sample
        samples.append(atoms.get_positions()[oxygen].copy())
        if len(samples) % 40 == 0:
            thermo.write(f'{done},{atoms.get_temperature():.1f},{atoms.get_potential_energy():.4f},{time.monotonic() - t0:.1f}\n')
            thermo.flush()
            np.save(out / 'o_positions.npy', np.array(samples))
            np.savez(ckpt, positions=atoms.get_positions(), momenta=atoms.get_momenta(), done=done)
    np.save(out / 'o_positions.npy', np.array(samples))
    np.savez(ckpt, positions=atoms.get_positions(), momenta=atoms.get_momenta(), done=done)
    positions = np.array(samples)
    times = np.arange(len(positions)) * args.sample * args.dt_fs / 1000.
    curve = msd(positions)
    summary = dict(structure=str(args.structure), key=args.key, temperature_K=args.temperature, dt_fs=args.dt_fs,
                   production_steps=done, production_ps=done * args.dt_fs / 1000., dtype=args.dtype,
                   seconds_this_invocation=time.monotonic() - begin,
                   ms_per_step=1000 * (time.monotonic() - t0) / max(1, done - (target - args.steps)))
    if len(positions) > 10:
        D, slope, intercept = diffusion_from_msd(times, curve)
        summary.update(D_O_cm2_s=D, msd_slope_A2_ps=slope, final_msd_A2=float(curve[-1]))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
