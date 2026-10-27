import unittest
from review05_gate import relative_gate

class RelativePrecision(unittest.TestCase):
    def test_common_offset_cancels(self):
        gate=relative_gate(-100,-101,-100.076,-101.076,.0065)
        self.assertTrue(gate['passed'])
        self.assertAlmostEqual(gate['absolute_relative_energy_error_eV_cell'],0)

    def test_config_dependent_energy_bias_fails(self):
        self.assertFalse(relative_gate(-100,-101,-100.076,-101.106,.001)['passed'])

    def test_force_error_independently_blocks(self):
        self.assertFalse(relative_gate(-100,-101,-100.076,-101.076,.020001)['passed'])

if __name__=='__main__':unittest.main()
