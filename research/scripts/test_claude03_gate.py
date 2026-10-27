import copy,unittest
from claude03_gate import p1_gate

class ReleaseGate(unittest.TestCase):
    def setUp(self):
        self.rows=[{'logical_job':name,'name':name,'accounting':{'State':'COMPLETED','ExitCode':'0:0','CPUTimeRAW':str(40*3600)},'vasp':{'SCF_converged':True}}
                   for name in ['03_L00_gamma','03b_L00_gamma_lrealauto','04_L01_gamma']]
        self.budget={'actual_plus_remaining_caps':2900,'p0_actual_plus_remaining_caps':210}

    def test_boundary_and_complete(self):
        self.assertTrue(p1_gate({'attempts':self.rows},self.budget)['allowed'])

    def test_claude04_explicit_hold_overrides_old_success_gate(self):
        for ref in ['REVIEW_CLAUDE_04_L00.md','REVIEW_CLAUDE_05_fast_gpu.md']:
            budget=dict(self.budget,approval_reference=ref)
            self.assertFalse(p1_gate({'attempts':self.rows},budget)['allowed'])

    def test_cost_or_scf_or_budget_blocks_release(self):
        cases=[]
        rows=copy.deepcopy(self.rows);rows[0]['accounting']['CPUTimeRAW']=str(40*3600+1);cases.append((rows,self.budget))
        rows=copy.deepcopy(self.rows);rows[2]['vasp']['SCF_converged']=False;cases.append((rows,self.budget))
        rows=copy.deepcopy(self.rows);rows[1]['accounting']['State']='TIMEOUT';cases.append((rows,self.budget))
        cases.append((self.rows[:2],self.budget))
        cases.append((self.rows,dict(self.budget,actual_plus_remaining_caps=3001)))
        cases.append((self.rows,dict(self.budget,p0_actual_plus_remaining_caps=217)))
        for rows,budget in cases:
            with self.subTest(rows=rows,budget=budget):self.assertFalse(p1_gate({'attempts':rows},budget)['allowed'])

if __name__=='__main__':unittest.main()
