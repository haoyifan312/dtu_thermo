import unittest

from RachfordRiceSolver import RachfordRiceSolverOption, RachfordRiceBase
from TwoPhaseFlash import *
from thermclc_interface import example_7_component, init_system
from SuccessiveSubstitutionSolver import *


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
            ss = TwoPhaseFlash(stream, solver)
            ss_default_solver = TwoPhaseFlash(stream)
            self.assertTrue(True)

    def test_case1(self):
        self._test_example_case(0)

    def test_case2(self):
        self._test_example_case(1)

    def test_case3(self):
        self._test_example_case(2)

    def test_case4(self):
        self._test_example_case(3)

    def test_case5(self):
        self._test_example_case(4)

    def test_question2(self):
        t = 205.0
        p = 6.0
        for i in range(10):
        # for i in (2,):
            p_new = p + i*0.02
            # p_new = 6.16
            self._test_case_for_t_p(t, p_new)

    def test_stability_analysis_restart_to_overshoot_case(self):
        t = 205.0
        p = 6.1
        self._test_case_for_t_p(t, p, .9292)

    def test_random_case(self):
        t = 205.0
        p = 6.124
        self._test_case_for_t_p(t, p)

    def _test_example_case(self, i):
        t = self.ts[i]
        p = self.ps[i]
        self._test_case_for_t_p(t, p, self.betas_gold[i])

    def _test_case_for_t_p(self, t, p, beta_gold=None):
        show_plot = False
        with init_system(self.components, 'SRK') as stream:
            ss = TwoPhaseFlash(stream)
            acc_by_cycle = SSAccelerationCriteriaByCycle(5)
            ss_acc_by_cycle = TwoPhaseFlash(stream, acceleration=SSAccelerationDEM(acc_by_cycle))
            acc_by_change = SSAccelerationCriteriaByChange(0.01)
            ss_acc_by_change = TwoPhaseFlash(stream, acceleration=SSAccelerationDEM(acc_by_change))

            ss_acc_by_change_sloppy = TwoPhaseFlash(stream,
                                                    rr_fast=RachfordRiceBase.create_solver(stream.inflow_size,
                                                                                                     RachfordRiceSolverOption.SLOPPY),
                                                    acceleration=SSAccelerationDEM(acc_by_change))
            flash_input = FlashInput(t, p, self.zs)
            iters, result, rr_iters = self.compute_successive_substitution_allow_max_iter_reached(flash_input,
                                                                                                          show_plot,
                                                                                                          ss)

            if beta_gold is not None:
                self.assertAlmostEqual(result.beta, beta_gold, 3)
            print(f'\nT={t} K; P={p} MPa')
            # print(result)
            iters_ac, result_a, rr_iters_ac = self.compute_successive_substitution_allow_max_iter_reached(flash_input,
                                                                                                          show_plot,
                                                                                                          ss_acc_by_cycle)

            iters_ac_byc, result_a_byc, rr_iters_ac_byc = self.compute_successive_substitution_allow_max_iter_reached(flash_input,
                                                                                                          show_plot,
                                                                                                          ss_acc_by_change)

            iters_sloppy, result_sloppy, rr_iters_sloppy = self.compute_successive_substitution_allow_max_iter_reached(flash_input,
                                                                                                          show_plot,
                                                                                                          ss_acc_by_change_sloppy)

            if beta_gold is not None:
                self.assertAlmostEqual(result_a.beta, result.beta, 3)
                self.assertAlmostEqual(result_a_byc.beta, result.beta, 3)
                self.assertAlmostEqual(result_sloppy.beta, result.beta, 3)
            print('\n\t\toriginal\tacc by cycle\tacc by change\tsloppy')
            print(f'beta\t{result.beta :.6f}\t{result_a.beta :.6f}\t{result_a_byc.beta :.6f}'
                  f'\t{result_sloppy.beta :.6f}')
            print(f'iters\t{iters}\t{iters_ac}\t{iters_ac_byc}\t{iters_sloppy}')
            print(f'rr iters\t{rr_iters}\t{rr_iters_ac}\t{rr_iters_ac_byc}\t{rr_iters_sloppy}')
            print(f'acc_count\t{ss._acceleration.counter}\t{ss_acc_by_cycle._acceleration.counter}\t'
                  f'{ss_acc_by_change._acceleration.counter}\t{ss_acc_by_change_sloppy._acceleration.counter}')
            print('\n\n\n')

    def compute_successive_substitution_allow_max_iter_reached(self, flash_input, show_plot, ss_acc_by_cycle):
        try:
            iters_ac, rr_iters_ac, result_a = ss_acc_by_cycle.compute(flash_input, show_plot=show_plot)
        except TwoPhaseFlashException as e:
            iters_ac = ss_acc_by_cycle._ss_max_iter
            result_a = e.result
            rr_iters_ac = e.total_rr_count
        return iters_ac, result_a, rr_iters_ac

