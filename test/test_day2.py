import math
import unittest

import numpy as np

from RachfordRiceSolver import RachfordRiceBase, RachfordRiceSolverOption
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
                self.assertEqual(pc, self.pc_gold[i] * 0.1013)
                self.assertEqual(omega, self.omega_gold[i])

    def test_wilson_k(self):
        t = 200
        p = 5.0
        with init_system(self.components, 'SRK') as stream:
            for i, comp in enumerate(self.components):
                k = stream.compute_wilson_k(t, p, i)
                print(f'Wilson K for {comp} at {t}K and {p}MPa = {k}')
                k_compare = compute_wilson_k(t, self.tc_gold[i], p, self.pc_gold[i] * 0.1013, self.omega_gold[i])
                self.assertAlmostEqual(k, k_compare)


class TestRachfordRiceSolver(unittest.TestCase):
    components = list(example_7_component.keys())
    zs = np.array(list(example_7_component.values()))
    p_mpa = 5.0
    ts = [190.0, 195, 200.0, 220.0, 280.0, 300.0, 350.0]
    beta_gold = [-0.1792, 0.3176, 0.7712, 0.9521, 0.9980, 1.0027, 1.0253]

    @property
    def size(self):
        return len(self.components)

    def create_solver(self, option=RachfordRiceSolverOption.BASE):
        return RachfordRiceBase.create_solver(self.size, option)

    def test_create_solver(self):
        solver = self.create_solver()
        self.assertEqual(solver.size, self.size)
        self.assertEqual(solver._result.beta, -1.0)
        self.assertTrue(np.allclose(solver._result.xs, np.zeros(self.size)))
        self.assertTrue(np.allclose(solver._result.ys, np.zeros(self.size)))

    def test_solve_liquid(self):
        t = self.ts[0]
        solver = self.create_solver()
        with init_system(self.components, 'SRK') as stream:
            ks = [stream.compute_wilson_k(t, self.p_mpa, i) for i in range(self.size)]
            result = solver.compute(ks, self.zs)
            print(result)
            self.assertEqual(result.beta, 0.0)
            self.assertTrue(np.allclose(result.xs, self.zs))
            self.assertTrue(np.allclose(result.ys, np.zeros(self.size)))

    def test_solve_vapor(self):
        t = self.ts[-1]
        solver = self.create_solver()
        with init_system(self.components, 'SRK') as stream:
            ks = [stream.compute_wilson_k(t, self.p_mpa, i) for i in range(self.size)]
            result = solver.compute(ks, self.zs)
            print(result)
            self.assertEqual(result.beta, 1.0)
            self.assertTrue(np.allclose(result.ys, self.zs))
            self.assertTrue(np.allclose(result.xs, np.zeros(self.size)))

    def test_solve_vle(self):
        i = 1
        t = self.ts[i]
        solver = self.create_solver()
        with init_system(self.components, 'SRK') as stream:
            ks = [stream.compute_wilson_k(t, self.p_mpa, i) for i in range(self.size)]
            result = solver.compute(ks, self.zs)
            print(result)
            self.assertAlmostEqual(result.beta, self.beta_gold[i], places=4)
            self.assertTrue(np.isclose(np.sum(result.xs), 1.0))
            self.assertTrue(np.isclose(np.sum(result.ys), 1.0))

    def test_solve_all(self):
        solver = self.create_solver()
        with init_system(self.components, 'SRK') as stream:
            for i, t in enumerate(self.ts):
                ks = [stream.compute_wilson_k(t, self.p_mpa, i) for i in range(self.size)]
                result = solver.compute(ks, self.zs)
                print(f'\nT={t}:')
                print(result)
                beta_gold = self.beta_gold[i]
                total_x = 1.0
                total_y = 1.0
                if beta_gold > 1.0:
                    beta_gold = 1.0
                    total_x = 0.0
                elif beta_gold < 0.0:
                    beta_gold = 0.0
                    total_y = 0.0
                self.assertAlmostEqual(result.beta, beta_gold, places=4)
                self.assertTrue(np.isclose(np.sum(result.xs), total_x))
                self.assertTrue(np.isclose(np.sum(result.ys), total_y))

    def test_convergence_pattern(self):
        """
        last few steps are quadratically converging
        """
        solver = self.create_solver()
        with init_system(self.components, 'SRK') as stream:
            for i in (1, 2, 3, 4):
                t = self.ts[i]
                ks = [stream.compute_wilson_k(t, self.p_mpa, i) for i in range(self.size)]
                result = solver.compute(ks, self.zs)
                betas = result.betas.copy()
                log_steps = []
                log_step_diffs = []
                betas = [float(num) for num in betas]
                print(len(betas))
                print(betas)
                print([betas[i]-betas[i-1] for i in range(1, len(betas))])
                for i in range(1, len(betas)):
                    this_beta = betas[i]
                    last_beta = betas[i - 1]
                    log_steps.append(math.log(abs(this_beta - last_beta)))
                    if len(log_steps) > 1:
                        log_step_diffs.append(log_steps[i - 1] - 2 * log_steps[i - 2])

                # print(log_steps)
                # print(log_step_diffs)

        with init_system(self.components, 'SRK') as stream:
            last_beta = 0.5
            for i in (1, 2, 3, 4):
                t = self.ts[i]
                ks = [stream.compute_wilson_k(t, self.p_mpa, i) for i in range(self.size)]
                result = solver.compute(ks, self.zs, initial_guess=last_beta)
                last_beta = result.beta
                betas = result.betas.copy()
                betas = [float(num) for num in betas]
                print(len(betas))
                print(betas)
                print([betas[i]-betas[i-1] for i in range(1, len(betas))])

    def test_solve_all_sloppy(self):
        solver = self.create_solver(RachfordRiceSolverOption.SLOPPY)
        beta_golds = np.array([0.0, 0.3505, 0.9749, 0.75, 0.75, 1.0, 1.0])
        with init_system(self.components, 'SRK') as stream:
            for i, t in enumerate(self.ts):
                ks = [stream.compute_wilson_k(t, self.p_mpa, i) for i in range(self.size)]
                result = solver.compute(ks, self.zs)
                print(f'\nT={t}:')
                print(result)
                total_x = 1.0
                total_y = 1.0
                beta_gold = beta_golds[i]
                if beta_gold == 1.0:
                    beta_gold = 1.0
                    total_x = 0.0
                elif beta_gold == 0.0:
                    beta_gold = 0.0
                    total_y = 0.0
                self.assertAlmostEqual(result.beta, beta_gold, places=3)
                self.assertTrue(np.isclose(np.sum(result.xs), total_x))
                self.assertTrue(np.isclose(np.sum(result.ys), total_y))

    def test_solve_all_negative_flash(self):
        solver = self.create_solver(RachfordRiceSolverOption.NEGATIVE_FLASH)
        with init_system(self.components, 'SRK') as stream:
            for i, t in enumerate(self.ts):
                ks = [stream.compute_wilson_k(t, self.p_mpa, i) for i in range(self.size)]
                result = solver.compute(ks, self.zs)
                print(f'\nT={t}:')
                print(result)
                beta_gold = self.beta_gold[i]
                total_x = 1.0
                total_y = 1.0
                if beta_gold > 1.0:
                    total_x = 0.0
                elif beta_gold < 0.0:
                    total_y = 0.0
                self.assertAlmostEqual(result.beta, beta_gold, places=4)
                # self.assertTrue(np.isclose(np.sum(result.xs), total_x))
                # self.assertTrue(np.isclose(np.sum(result.ys), total_y))
