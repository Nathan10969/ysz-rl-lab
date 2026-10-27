"""Claude05 precision criterion: errors of same-composition energy differences, plus forces."""
def relative_gate(ref_L00,ref_L01,fast_L00,fast_L01,max_force_difference_eV_A):
    ref_delta=ref_L01-ref_L00
    fast_delta=fast_L01-fast_L00
    error=abs(fast_delta-ref_delta)
    return {'reference_relative_energy_eV':ref_delta,'fast_relative_energy_eV':fast_delta,
            'absolute_relative_energy_error_eV_cell':error,
            'max_atom_force_difference_eV_A':max_force_difference_eV_A,
            'passed':error<=.02 and max_force_difference_eV_A<=.02}
