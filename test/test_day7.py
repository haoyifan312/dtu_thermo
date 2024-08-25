import unittest

import numpy as np
from matplotlib import pyplot as plt

from RachfordRiceSolver import *
from MultiPhaseRachfordRice import *
from StabilityAnalysis import StabilityAnalysis
from SuccessiveSubstitutionSolver import SSAccelerationDEM, SSAccelerationCriteriaByChange, \
    SSAccelerationCriteriaByCycle
from thermclc_interface import init_system, PropertyType


class TestMultiPhaseRachfordRice(unittest.TestCase):
    inflows = {
        'C1': 0.66,
        'C2': 0.03,
        'C3': 0.01,
        'CO2': 0.05,
        'H2S': 0.25
    }
    t = 201
    ts = [196, 198, 200, 201, 203, 204, 205]
    p = 4.0
    stream = ThermclcInterface(list(inflows.keys()), 'SRK')
    n_phases = 3
    inflow_map = {inflow: i for i, inflow in enumerate(inflows.keys())}

    initial_betas_gold = [[0.0125, 0.9875, 0],
                          [0.0125, 0.9875, 0],
                          [0.0138, 0.984, 0.0021],
                          [0.1574, 0.6135, 0.229],
                          [0.3266, 0.2012, 0.4722],
                          [0.3844, 0.07, 0.5456],
                          [0.4151, 0, 0.5849]]

    final_beta_gold = [[0.166, 0.7601, 0.0739],
                       [0.2063, 0.4562, 0.3375],
                       [0.2421, 0.2849, 0.473],
                       [0.2641, 0.219, 0.5168],
                       [0.3657, 0.0565, 0.5779],
                       [0.4078, 0, 0.5922],
                       [0.3983, 0, 0.6017]]

    def test_constructor(self):
        mrr = self.create_mrr()

    def test_equal_molar_with_initial_guess(self):
        mrr = self.create_mrr()
        self.setup_mrr_with_inflow_and_phi_from_wilson(mrr, self.t, self.p)

        mrr._update_Ei()
        mrr_q = mrr.compute_q()
        print(f'Q={mrr_q}')
        self.assertAlmostEqual(mrr_q, 0.18899850045870425)

        mrr._update_gradient()
        print(f'g={mrr._gradient_k}')
        self.assertTrue(np.allclose(mrr._gradient_k, np.array([0.07379915, -0.01630875, -0.0574904])))

        mrr._update_hessian()
        print(f'h={mrr._hessian_kl}')
        self.assertTrue(np.allclose(mrr._hessian_kl, np.array([[1.46626628, 0.88703858, 0.42529769],
                                                               [0.88703858, 1.06280731, 1.09908035],
                                                               [0.42529769, 1.09908035, 1.64809318]])))

    def test_q_minimization(self):
        mrr = self.create_mrr()
        self.setup_mrr_with_inflow_and_phi_from_wilson(mrr, self.t, self.p)

        q, iters = mrr.minimize_q()
        beta = mrr.get_effective_beta()
        print(f'beta={beta}')
        print(f'g={mrr._gradient_k}')
        print(f'iters={iters}')
        print(f'q={q}')
        self.assertTrue(np.allclose(beta, np.array([0.15741053, 0.61353954, 0.22904993])))
        self.assertAlmostEqual(q, 0.18304988463622784)
        self.assertTrue(iters < 5)

    def test_q_minimization_remove_phase(self):
        mrr = self.create_mrr()
        self.setup_mrr_with_inflow_and_phi_from_wilson(mrr, 198.0, 4.0)

        q, iters = mrr.minimize_q()
        beta = mrr.get_effective_beta()
        print(f'beta={beta}')
        print(f'g={mrr._gradient_k}')
        print(f'iters={iters}')
        print(f'q={q}')
        self.assertTrue(np.allclose(beta, np.array([0.01251599, 0.98748401, 0.0])))
        self.assertAlmostEqual(q, 0.08003948041420372)
        self.assertTrue(iters < 6)

    def test_all_q_minimization_for_initial_guess(self):
        mrr = self.create_mrr()
        for i, t in enumerate(self.ts):
            self.setup_mrr_with_inflow_and_phi_from_wilson(mrr, t, self.p)

            print(f'\n\nT={t}')
            mrr.set_all_phase_active()
            q, iters = mrr.minimize_q()
            beta = mrr.get_effective_beta()
            print(f'beta={beta}')
            print(f'g={mrr._gradient_k}')
            print(f'iters={iters}')
            print(f'q={q}')
            self.assertTrue(np.allclose(beta, np.array(self.initial_betas_gold[i]), atol=1e-4))

    def test_ss_201(self):
        i = 3
        ss_mrr = self.create_ss_mrr()
        self._test_ss_mrr_for_case(i, ss_mrr)

    def test_ss_196_no_initial_vap(self):
        i = 0
        ss_mrr = self.create_ss_mrr()
        self._test_ss_mrr_for_case(i, ss_mrr)

    def test_ss_205_no_l1(self):
        i = 6
        ss_mrr = self.create_ss_mrr()
        self._test_ss_mrr_for_case(i, ss_mrr)

    def test_all_ss_cases(self):
        ss_mrr = self.create_ss_mrr()
        ss_mrr_acc = self.create_ss_mrr(acceleration=SSAccelerationDEM(SSAccelerationCriteriaByCycle(5)))
        for i in range(len(self.ts)):
            print('Without successive substitution acceleration')
            self._test_ss_mrr_for_case(i, ss_mrr)
            print('With successive substitution acceleration')
            self._test_ss_mrr_for_case(i, ss_mrr_acc)

    def test_ss_to_find_3_phase_t_range(self):
        """ 195.64; 203.3"""
        t_end = 210
        t_start = 190
        steps = 50
        step_size = (t_end-t_start)/steps
        ss_mrr = self.create_ss_mrr()
        beta_l1 = []
        beta_l2 = []
        beta_v = []
        ts = []
        for i in range(steps + 1):
            t = t_start + step_size * i
            ts.append(t)
            betas = self._test_ss_mrr_for_tp(ss_mrr, t, self.p)
            each_beta_l1, each_beta_l2, each_beta_v = betas
            beta_l1.append(each_beta_l1)
            beta_l2.append(each_beta_l2)
            beta_v.append(each_beta_v)

        plt.plot(ts, beta_l1, label='CH4-rich liquid phase')
        plt.plot(ts, beta_l2, label='H2S-rich liquid phase')
        plt.plot(ts, beta_v, label='vapor phase')
        plt.legend()
        plt.xlabel('Temperature (K)')
        plt.ylabel(r'$\beta$')
        plt.ylim((0.0, 1.0))
        plt.savefig('multiphase beta.png')

    def _test_ss_mrr_for_case(self, i, ss_mrr):
        t = self.ts[i]
        p = self.p
        return self._test_ss_mrr_for_tp(ss_mrr, t, p, self.final_beta_gold[i])

    def _test_ss_mrr_for_tp(self, ss_mrr, t, p, gold=None):
        l1_ln_phi = self.get_l1_ln_phi(t, p)
        l2_ln_phi = self.get_l2_ln_phi(t, p)
        initial_phi = np.array([np.exp(l1_ln_phi),
                                np.exp(l2_ln_phi),
                                np.ones(len(self.inflows))])
        beta, iters_ss, iters_newton = ss_mrr.solve(t, p, self.inflow_moles, np.transpose(initial_phi))
        print(f'T={t}')
        print(f'beta={beta}')
        print(f'ss iters={iters_ss}')
        print(f'newton iters={iters_newton}\n\n')
        if gold is not None:
            self.assertTrue(np.allclose(beta, gold, atol=1e-4))
        return beta

    def test_stability(self):
        for i in range(len(self.ts)):
            self._test_stability_for_case(i)

    def _test_stability_for_case(self, i):
        t = self.ts[i]
        p = self.p
        print(f'\nT={t}\tP={p}')

        print('test stability with H2S pure liquid guess')
        wi_h2s = np.zeros(len(self.inflow_names))
        wi_h2s[self.inflow_names.index('H2S')] = 1.0
        self._test_stability_for_tp(self.stream, FlashInput(t, p, self.inflow_moles), wi_h2s)

        print('test stability with CH4 pure liquid guess')
        wi_h2s = np.zeros(len(self.inflow_names))
        wi_h2s[self.inflow_names.index('C1')] = 1.0
        self._test_stability_for_tp(self.stream, FlashInput(t, p, self.inflow_moles), wi_h2s)

        print('test stability with ideal gas guess')
        ki = self.stream.all_wilson_ks(t, p, PropertyType.PROPERTY)
        rr_solver = RachfordRiceBase.create_solver(len(self.inflow_names), RachfordRiceSolverOption.BASE)
        result = rr_solver.compute(ki, self.inflow_moles)
        self._test_stability_for_tp(self.stream, FlashInput(t, p, self.inflow_moles), result.ys_or_zs)

    def _test_stability_for_tp(self, stream: ThermclcInterface, flash_input: FlashInput, estimate_wi):
        acc_by_cycle = SSAccelerationCriteriaByCycle(5)
        acc = SSAccelerationDEM(acc_by_cycle)
        sa = StabilityAnalysis(stream, acceleration=acc)
        sa_result, ss_iters = sa.compute(flash_input, estimate_wi)
        print(f'Stability analysis: tm={sa_result.distance}, iters={ss_iters}')
        print(f'Stability analysis: wi={sa_result.wi}\n\n')

    def setup_mrr_with_inflow_and_phi_from_wilson(self, mrr, t, p):
        mrr.set_zi(self.inflow_moles)
        mrr.set_phi_for_phase(np.ones(mrr.component_size), MultiPhaseIndexVLLE.VAPOR)
        # c1-rich liquid phase
        l1_ln_phi = self.get_l1_ln_phi(t, p)
        mrr.set_phi_for_phase(np.exp(l1_ln_phi), MultiPhaseIndexVLLE.LIQUID1)
        # h2s-rich liquid phase
        l2_ln_phi = self.get_l2_ln_phi(t, p)
        mrr.set_phi_for_phase(np.exp(l2_ln_phi), MultiPhaseIndexVLLE.LIQUID2)
        equal_molar_beta = np.array([1.0 / self.n_phases] * self.n_phases)
        mrr.set_beta(equal_molar_beta)

    def get_l2_ln_phi(self, t, p):
        l2_ln_phi = self.get_all_ln_phi_assuming_ideal_vapor(t, p)
        l2_ln_phi[self.inflow_map['H2S']] = self.get_h2s_ln_phi_in_rich_liquid(t, p)
        return l2_ln_phi

    def get_l1_ln_phi(self, t, p):
        l1_ln_phi = self.get_all_ln_phi_assuming_ideal_vapor(t, p)
        l1_ln_phi[self.inflow_map['C1']] = self.get_c1_ln_phi_in_rich_liquid(t, p)
        return l1_ln_phi

    def get_c1_ln_phi_in_rich_liquid(self, t, p):
        return self.get_c1_ln_phi_in_lean_liquid(t, p) + 1.0

    def get_c1_ln_phi_in_lean_liquid(self, t, p):
        return self.get_component_ln_k(t, p, 'C1')

    def get_component_ln_k(self, t, p, component):
        kw = self.stream.compute_wilson_k(t, p, component)
        return np.log(kw)

    def get_h2s_ln_phi_in_lean_liquid(self, t, p):
        return self.get_component_ln_k(t, p, 'H2S')

    def get_h2s_ln_phi_in_rich_liquid(self, t, p):
        return self.get_h2s_ln_phi_in_lean_liquid(t, p) + 1.0

    def get_all_ln_phi_assuming_ideal_vapor(self, t, p):
        all_k_w = self.stream.all_wilson_ks(t, p, PropertyType.PROPERTY)
        return np.log(all_k_w)

    def create_mrr(self):
        return MultiPhaseRachfordRice(self.stream, self.n_phases)

    def create_ss_mrr(self, acceleration=None):
        return SuccessiveSubstitutionForMRR(self.stream, self.n_phases, acceleration=acceleration)

    @property
    def inflow_names(self):
        return list(self.inflows.keys())

    @property
    def inflow_moles(self):
        return np.array(list(self.inflows.values()))
