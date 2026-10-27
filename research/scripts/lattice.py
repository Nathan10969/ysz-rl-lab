"""Ideal fluorite site occupations; X is a bookkeeping vacancy, never an MLIP atom."""
from functools import lru_cache
import numpy as np
import spglib
from ase.spacegroup import crystal


def reference_lattice():
    return crystal(['Zr', 'O'], basis=[(0,0,0),(.25,.25,.25)], spacegroup=225,
                   cellpar=[5.14,5.14,5.14,90,90,90]).repeat((2,2,2))


def make_state(seed, n_y=4, dopant='Y'):
    if n_y % 2 or not 0 < n_y < 32:
        raise ValueError('Use an even, positive dopant count below 32.')
    state = reference_lattice()
    rng = np.random.default_rng(seed)
    cats = np.flatnonzero(state.numbers == 40)
    oxy = np.flatnonzero(state.numbers == 8)
    for i in rng.choice(cats, n_y, replace=False):
        state[int(i)].symbol = dopant
    state.numbers[rng.choice(oxy, n_y//2, replace=False)] = 0
    return state


def physical_atoms(state):
    atoms = state[state.numbers != 0]
    atoms.set_array('site_id', np.flatnonzero(state.numbers != 0))
    return atoms


@lru_cache(maxsize=1)
def permutations():
    parent = reference_lattice()
    frac = parent.get_scaled_positions()
    sym = spglib.get_symmetry((parent.cell.array, frac, parent.numbers), symprec=1e-5)
    out = []
    for rot, trans in zip(sym['rotations'], sym['translations']):
        moved = (frac @ rot.T + trans) % 1
        delta = moved[:,None,:] - frac[None,:,:]
        delta -= np.round(delta)
        d2 = np.sum(delta**2, axis=2)
        perm = np.argmin(d2, axis=1)
        if np.max(d2[np.arange(len(parent)),perm]) > 1e-10 or len(set(perm)) != len(parent):
            raise ValueError('Nonbijective parent-lattice symmetry mapping.')
        out.append(perm)
    return np.unique(out, axis=0)


def canonical_key(state):
    inv = np.argsort(permutations(), axis=1)
    return min(row.astype(np.uint8).tobytes() for row in state.numbers[inv]).hex()


def apply_action(state, action):
    op = action['op']
    i,j = action['sites']
    if not isinstance(i,int) or not isinstance(j,int) or i == j or min(i,j)<0 or max(i,j)>=len(state):
        raise ValueError('Invalid site ids.')
    pair = {int(state.numbers[i]), int(state.numbers[j])}
    if (op == 'swap_cations' and pair != {39,40}) or (op == 'move_vacancy' and pair != {0,8}):
        raise ValueError('Action does not match the occupied sublattice.')
    if op not in ['swap_cations','move_vacancy']:
        raise ValueError('Unknown operation.')
    moved = state.copy()
    moved.numbers[i], moved.numbers[j] = moved.numbers[j], moved.numbers[i]
    return moved
