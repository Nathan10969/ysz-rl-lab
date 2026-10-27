"""Review12 L1 (10.2 mol% Y2O3: Zr88Y20O206, 314 atoms): MACE check and zero-shot blind rollouts.

The v2.2 environment fixes the 8 mol% composition in two places
(claude08_env.COMPOSITION and _validate_physical_geometry). This entry point
patches only those, plus the start pools, at runtime; every other rule
(actions, relaxation protocol, mapping, rewards, constrained decoding) is
unchanged. No model is trained here.

Stages:
  l1calib  16 L1 terminals (8 relaxed random starts + 8 short SA chains) for the preregistered
           MACE-vs-DFT gate (Spearman >= 0.7; >= 2 of MACE top-3 in DFT top-6; force RMSE median
           <= 0.20 eV/A, per-atom P95 <= 0.50 eV/A), DFT single points made by make_dft.py.
  blind    as formal12_rl --stage blind, on 4 sealed L1 starts (132 relaxations per arm/seed).
"""
from __future__ import annotations
import sys
from collections import Counter

import numpy as np

import claude08_env as env
import formal12_common as common

L1_COMPOSITION = {0: 10, 8: 206, 39: 20, 40: 88}
L1_STARTS = dict(blind=4, calib=16)


def _validate_physical_geometry_l1(atoms):
    parent = env._parent()
    if len(atoms) != 314 or dict(zip(*np.unique(atoms.numbers, return_counts=True))) != {8: 206, 39: 20, 40: 88}:
        raise env.MappingError("Relaxed physical composition must be Zr88Y20O206")
    if not np.all(atoms.pbc) or not np.allclose(atoms.cell.array, parent.cell.array, rtol=0, atol=1e-10):
        raise env.MappingError("Fixed periodic mother cell changed")
    if not np.all(np.isfinite(atoms.positions)):
        raise env.MappingError("Nonfinite relaxed positions")


def start_pools_l1(seed):
    from calibration import make_state
    rng = common.stream(seed, 'starts-L1')
    keys, occupancies = set(), []
    while len(occupancies) < sum(L1_STARTS.values()):
        occupancy = env.validate_occupancy(make_state(int(rng.integers(0, 2**63 - 1)), 3, 20).numbers)
        key = env.occupancy_key(occupancy)
        if key in keys: continue
        keys.add(key); occupancies.append(list(occupancy))
    return dict(blind=occupancies[:4], calib=occupancies[4:20])


def apply_l1():
    import inspect
    original = env._validate_physical_geometry
    env.COMPOSITION = dict(L1_COMPOSITION)
    env._validate_physical_geometry = _validate_physical_geometry_l1
    # map_terminal checks a one-to-one assignment of exactly 316 atoms; use the actual atom count.
    source = inspect.getsource(env.map_terminal)
    if source.count('!= 316') != 1: raise RuntimeError('Unexpected map_terminal source; refusing to patch')
    exec(compile(source.replace('!= 316', '!= len(atoms)'), env.__file__, 'exec'), env.__dict__)
    import formal12_rl
    formal12_rl.start_pools = start_pools_l1
    formal12_rl.BLIND_RELAX_CAP = L1_STARTS['blind'] * (1 + common.GROUP_SIZE * common.K)
    return original


def job_l1calib(args, ledger, out):
    import formal12_search as search
    from formal12_rl import terminals_of, terminal_rows
    pools = start_pools_l1(args.seed)
    engine = common.make_physics(args.checkpoint, out, args.device)
    cache, namespace = env.RequestCache(), ('l1calib', args.seed)
    starts = pools['calib']
    initials = search.relax_initials(engine, cache, namespace, starts, ledger, log=out / 'physics.jsonl')
    chains = search.run_chains([('SA', 8 + i, initials[8 + i]) for i in range(8)], engine, cache, namespace, ledger,
                               rng_label='l1calib', seed=args.seed, log=out / 'physics.jsonl')
    picked = [initials[i] for i in range(8)] + [c.current for c in chains]
    rows = terminal_rows(terminals_of(cache, namespace))
    by_key = {r['terminal_key']: r for r in rows}
    keys = []
    for terminal in picked:
        key = terminal.mapping.canonical_key
        if key not in keys: keys.append(key)
    forces = {t.mapping.canonical_key: t.result.forces.tolist() for t in picked}
    for key in keys: by_key[key]['mace_forces_eV_A'] = forces[key]
    return dict(terminals=[by_key[k] for k in keys], diagnostics=keys, composition='Zr88Y20O206',
                merged_terminal_keys=keys, chain_summaries=[search.chain_summary(c) for c in chains])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    apply_l1()
    import formal12_rl
    if '--stage' in argv and argv[argv.index('--stage') + 1] == 'l1calib':
        argv[argv.index('--stage') + 1] = 'search'      # reuse the CLI/ledger; dispatch below
        formal12_rl.job_search = job_l1calib
        formal12_rl.relax_cap = lambda args: 16 + 8 * common.K
    formal12_rl.main(argv)


if __name__ == '__main__':
    main()
