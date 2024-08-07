import unittest

import numpy as np

from RachfordRiceSolver import RachfordRiceSolverOption, RachfordRiceBase
from SuccessiveSubstitutionSolver import SuccessiveSubstitutionSolver, FlashInput
from thermclc_interface import example_7_component, init_system


class TestSuccessiveSubstitution(unittest.TestCase):
    ts = [200.0, 205.0, 220.0, 220.0, 203.0]
    ps = [5.0, 5.0, 5.0, 7.0, 5.6]
    components = list(example_7_component.keys())
    zs = np.array(list(example_7_component.values()))
    betas_gold = [0.8283, 0.9319, 0.9781, 0.9809, 0.8047]

    @property
    def size(self):
        return len(self.components)

    def create_solver(self, option=RachfordRiceSolverOption.BASE):
        return RachfordRiceBase.create_solver(self.size, option)

    def test_wilson_k_in_vle_region(self):
        solver = self.create_solver()
        gold = [True, True, True, True, True]
        with init_system(self.components, 'SRK') as stream:
            for i, (t, p) in enumerate(zip(self.ts, self.ps)):
                ks = [stream.compute_wilson_k(t, p, i) for i in range(self.size)]
                solver._g_solver.set_input(ks, self.zs)
                initial_guess_vle = solver._g_solver.is_vle()
                self.assertEqual(gold[i], initial_guess_vle)

    def test_constructor(self):
        solver = self.create_solver()
        with init_system(self.components, 'SRK') as stream:
            ss = SuccessiveSubstitutionSolver(stream, solver)
            ss_default_solver = SuccessiveSubstitutionSolver(stream)
            self.assertTrue(True)

    def test_case1(self):
        self._test_example_case(0)

    def test_case2(self):
        self._test_example_case(1)

    def test_case3(self):
        self._test_example_case(2)

    def test_case4(self):
        self._test_example_case(3)

    def test_case5_no_split_from_wilson_ks(self):
        self._test_example_case(4, initial_ks=None)

    def _test_example_case(self, i, initial_ks=None):
        t = self.ts[i]  # 200 K
        p = self.ps[i]  # 5 MPa
        with init_system(self.components, 'SRK') as stream:
            ss = SuccessiveSubstitutionSolver(stream)
            iters, result = ss.compute(FlashInput(t, p, self.zs), initial_ks=initial_ks)
            print(f'T={t} K; P={p} MPa')
            print(f'total successive substitution iterations={iters}')
            print(result)
            self.assertAlmostEqual(result.beta, self.betas_gold[i], 3)
