"""CPU checks for the user-approved batch1/4/16 throughput revision."""
import unittest

import smoke_mace


class MaceProtocolTests(unittest.TestCase):
    def test_real_atomic_data_batches_do_not_require_constructor_without_fields(self):
        import torch
        from types import SimpleNamespace
        from ase.io import read
        from mace.tools.utils import AtomicNumberTable
        self.assertTrue(callable(getattr(smoke_mace, 'build_batch', None)),
                        'A direct AtomicData.from_config batch builder is required')
        calc = SimpleNamespace(charges_key='Qs', arrays_keys={}, info_keys={}, head='Default',
                               available_heads=['Default'], z_table=AtomicNumberTable([8, 39, 40]), r_max=5.)
        atoms = read(smoke_mace.SMOKE / 'data/mace_initial_physical.extxyz', index=0)
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            for size in (1, 4, 16):
                batch = smoke_mace.build_batch(calc, [atoms.copy() for _ in range(size)], 'cpu')
                self.assertEqual(batch.num_graphs, size)
                self.assertEqual(tuple(batch.positions.shape), (316 * size, 3))
                self.assertEqual(batch.positions.dtype, torch.float64)
                self.assertEqual(batch.ptr.tolist(), [316*i for i in range(size+1)])
        finally:
            torch.set_default_dtype(previous)

    def test_batch16_has_sixteen_graphs_without_claiming_sixteen_unique_inputs(self):
        ids = smoke_mace.batch_indices(16, 8)
        self.assertEqual(len(ids), 16)
        self.assertEqual(len(set(ids)), 8)
        self.assertEqual(ids, list(range(8)) * 2)

    def test_all_three_relaxation_batch_sizes_fit_original_raw_call_cap(self):
        cases = smoke_mace.relaxation_cases(8)
        self.assertEqual([len(ids) for _, ids in cases], [1, 4, 16])
        self.assertEqual(smoke_mace.raw_ef_worst_case(), 1023)
        self.assertLessEqual(smoke_mace.raw_ef_worst_case(), 1024)
        self.assertEqual(smoke_mace.LIMITS['wall_seconds'], 1800)
        self.assertEqual(smoke_mace.LIMITS['relax_max_steps'], 40)


if __name__ == '__main__':
    unittest.main()
