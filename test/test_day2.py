import unittest
from thermclc_interface import *


class TestWilsonK(unittest.TestCase):
    components = ['C1', 'C3']
    tc_gold = [190.6, 369.8]
    pc_gold = [45.4, 41.9]
    omega_gold = [0.008, 0.152]

    def test_critical_properties(self):
        with init_system(self.components, 'SRK') as stream:
            for i, comp in enumerate(self.components):
                tc, pc, omega = stream.get_critical_properties(i)
                # print(f'{comp}: TC={tc}\tPC={pc}\t\tOmega={omega}')
                self.assertEqual(tc, self.tc_gold[i])
                self.assertEqual(pc, self.pc_gold[i]*0.1013)
                self.assertEqual(omega, self.omega_gold[i])

    def test_wilson_k(self):
        t = 200
        p = 5.0
        with init_system(self.components, 'SRK') as stream:
            for i, comp in enumerate(self.components):
                k = stream.compute_wilson_k(t, p, i)
                print(f'Wilson K for {comp} at {t}K and {p}MPa = {k}')
                k_compare = compute_wilson_k(t, self.tc_gold[i], p, self.pc_gold[i]*0.1013, self.omega_gold[i])
                self.assertAlmostEqual(k, k_compare)


class TestRachfordRiceSolver(unittest.TestCase):
    components = list(example_7_component.keys())
    z = np.array(list(example_7_component.values()))
    p_mpa = 5.0

    @property
    def size(self):
        return len(self.components)

    def test_create_solver(self):
        solver = RachfordRiceBase(self.size)



