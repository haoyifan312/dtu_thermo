import unittest

import numpy as np

from MultiPhaseRachfordRice import *
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
    p = 4.0
    stream = ThermclcInterface(list(inflows.keys()), 'SRK')
    n_phases = 3
    inflow_map = {inflow: i for i, inflow in enumerate(inflows.keys())}

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
        self.assertTrue(iters < 5)

    def setup_mrr_with_inflow_and_phi_from_wilson(self, mrr, t, p):
        mrr.set_zi(self.inflow_moles)
        mrr.set_phi_for_phase(np.ones(mrr.component_size), MultiPhaseIndexVLLE.VAPOR)
        # c1-rich liquid phase
        l1_ln_phi = self.get_all_ln_phi_assuming_ideal_vapor(t, p)
        l1_ln_phi[self.inflow_map['C1']] = self.get_c1_ln_phi_in_rich_liquid(t, p)
        mrr.set_phi_for_phase(np.exp(l1_ln_phi), MultiPhaseIndexVLLE.LIQUID1)
        # h2s-rich liquid phase
        l2_ln_phi = self.get_all_ln_phi_assuming_ideal_vapor(t, p)
        l2_ln_phi[self.inflow_map['H2S']] = self.get_h2s_ln_phi_in_rich_liquid(t, p)
        mrr.set_phi_for_phase(np.exp(l2_ln_phi), MultiPhaseIndexVLLE.LIQUID2)
        equal_molar_beta = np.array([1.0 / self.n_phases] * self.n_phases)
        mrr.set_beta(equal_molar_beta)

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

    @property
    def inflow_names(self):
        return list(self.inflows.keys())

    @property
    def inflow_moles(self):
        return np.array(list(self.inflows.values()))
