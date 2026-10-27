import unittest
import numpy as np
from lattice import reference_lattice, make_state, physical_atoms, canonical_key, permutations, apply_action

class LatticeIntegrity(unittest.TestCase):
    def test_charge_compensated_ysz(self):
        ref = reference_lattice()
        state = make_state(17)
        atoms = physical_atoms(state)
        self.assertEqual(len(ref), 96)
        self.assertEqual(len(atoms), 94)
        counts = {s: state.get_chemical_symbols().count(s) for s in ['Zr', 'Y', 'O', 'X']}
        self.assertEqual(counts, dict(Zr=28, Y=4, O=62, X=2))
        self.assertEqual(4*counts['Zr']+3*counts['Y']-2*counts['O'], 0)

    def test_symmetry_key_is_invariant_but_not_constant(self):
        state = make_state(17)
        key = canonical_key(state)
        perms = permutations()
        self.assertGreater(len(perms), 100)
        for perm in perms[::max(1, len(perms)//10)]:
            rotated = state.copy()
            numbers = state.numbers.copy()
            rotated.numbers[perm] = numbers
            self.assertEqual(key, canonical_key(rotated))
        self.assertNotEqual(key, canonical_key(make_state(18)))

    def test_moves_preserve_species_and_reject_wrong_sublattice(self):
        state = make_state(17)
        y = int(np.where(state.numbers == 39)[0][0]);zr = int(np.where(state.numbers == 40)[0][0])
        v = int(np.where(state.numbers == 0)[0][0]);o = int(np.where(state.numbers == 8)[0][0])
        for action in [{'op':'swap_cations','sites':[y,zr]}, {'op':'move_vacancy','sites':[v,o]}]:
            moved = apply_action(state, action)
            self.assertEqual(sorted(state.numbers), sorted(moved.numbers))
        with self.assertRaises(ValueError):
            apply_action(state, {'op':'swap_cations','sites':[y,o]})

if __name__ == '__main__':
    unittest.main()
