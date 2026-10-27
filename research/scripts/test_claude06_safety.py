"""Regression tests for spending caps and acceptance of complete VASP outputs."""
import unittest
from claude06_safety import budget_projection, validate_output
from render_claude06_report import gate


class SafetyTests(unittest.TestCase):
    def test_running_spending_is_not_double_counted(self):
        jobs=[{'name':'a','cap':4},{'name':'b','cap':4},{'name':'c','cap':.25}]
        observed={'a':{'state':'RUNNING','actual':1},'b':{'state':'FAILED','actual':.02}}
        result=budget_projection(jobs,observed,historical=.0741666667)
        self.assertAlmostEqual(result['actual'],1.0941666667)
        self.assertAlmostEqual(result['remaining_caps'],3.25)
        self.assertAlmostEqual(result['actual_plus_remaining_caps'],4.3441666667)

    def test_full_plan_includes_historical_gpu_failures(self):
        jobs=[{'name':str(i),'cap':.25} for i in range(16)]
        jobs += [{'name':'relax'+str(i),'cap':239/60} for i in range(9)]
        self.assertLess(budget_projection(jobs,{},.0741666667)['actual_plus_remaining_caps'],40)

    def test_ionic_nonconvergence_is_not_a_valid_relaxation(self):
        xml='<modeling><parameters><i name="NELM">120</i></parameters><calculation><scstep/></calculation></modeling>'
        out='aborting loop because EDIFF is reached'
        self.assertTrue(validate_output(xml,out,False)['valid'])
        self.assertFalse(validate_output(xml,out,True)['valid'])
        self.assertTrue(validate_output(xml,out+'\nreached required accuracy - stopping structural energy minimisation',True)['valid'])

    def test_one_failed_electronic_step_invalidates_relaxation(self):
        xml='<modeling><parameters><i name="NELM">2</i></parameters><calculation><scstep/><scstep/></calculation><calculation><scstep/></calculation></modeling>'
        out='aborting loop because EDIFF is reached\nreached required accuracy - stopping structural energy minimisation'
        self.assertFalse(validate_output(xml,out,True)['valid'])

    def test_good_energy_ranking_cannot_hide_bad_forces(self):
        rows=[dict(config=str(i),MACE_energy_eV=i,DFT_energy_eV=2*i,
                   force_component_RMSE_eV_A=.01,atom_force_errors_eV_A=[.01]*10) for i in range(16)]
        self.assertTrue(gate(rows,16)['passed'])
        self.assertIsNone(gate(rows[:15],16)['passed'])
        for row in rows:row['force_component_RMSE_eV_A']=.3
        self.assertFalse(gate(rows,16)['passed'])


if __name__=='__main__':unittest.main()
