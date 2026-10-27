import unittest
import numpy as np
from calibration import reference, make_state, canonical_key, job_specs

class CalibrationInputs(unittest.TestCase):
    def test_large_cell_is_8mol_and_neutral(self):
        s=make_state(20261023,3,16)
        c={x:s.get_chemical_symbols().count(x) for x in ['Zr','Y','O','X']}
        self.assertEqual(c,{'Zr':92,'Y':16,'O':208,'X':8})
        self.assertEqual(4*c['Zr']+3*c['Y']-2*c['O'],0)
        self.assertEqual(len(s[s.numbers!=0]),316)
        self.assertAlmostEqual(c['Y']/(2*108-c['Y']),.08)

    def test_key_is_invariant_to_parent_translation(self):
        s=make_state(20261023,3,16)
        transformed=s.copy()
        frac=s.get_scaled_positions();shifted=(frac+np.array([1/3,0,0]))%1
        delta=shifted[:,None,:]-frac[None,:,:];delta-=np.round(delta)
        mapping=np.argmin(np.sum(delta**2,axis=2),axis=1)
        transformed.numbers[mapping]=s.numbers
        self.assertEqual(canonical_key(s,3),canonical_key(transformed,3))
        self.assertNotEqual(canonical_key(s,3),canonical_key(make_state(20261024,3,16),3))

    def test_all_jobs_fit_authorized_remainder_and_are_small_allocations(self):
        jobs=job_specs()
        self.assertEqual(len({j['id'] for j in jobs}),len(jobs))
        self.assertLess(sum(j['cores']*j['limit_hours'] for j in jobs)+16*23/3600,3000)
        self.assertTrue(all(j['cores'] in [16,32] for j in jobs))
        # Both actual main-cell arrangements need matched Gamma / 2x2x2 calculations.
        for cid in ['L00','L01']:
            self.assertEqual({tuple(j['kmesh']) for j in jobs if j['config']==cid},{(1,1,1),(2,2,2)})
        self.assertEqual(len(jobs),31)
        self.assertEqual(sum(j['hard_core_hours'] for j in jobs),2888)
        p0=[j for j in jobs if j['phase']=='P0_cost']
        self.assertEqual(len(p0),5)
        self.assertEqual(sum(j['hard_core_hours'] for j in p0),216)
        by_id={j['id']:j for j in jobs}
        self.assertEqual(by_id['03b_L00_gamma_lrealauto']['lreal'],'Auto')
        self.assertEqual({j['config'] for j in jobs if j['phase']=='P4_small_relax'},
                         {'S00','S01','S04','S08'})
        for cid,number in [('S02',31),('S06',32)]:
            self.assertEqual(by_id[f'{number}_{cid}_relax_fromideal']['input_geometry'],
                             'ideal_initial_occupation')

if __name__=='__main__':unittest.main()
