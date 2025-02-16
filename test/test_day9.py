import json
import unittest

import numpy as np
from matplotlib import pyplot as plt

from ReactionSystem import ComponentType, Component, ReactionSystem


class TestReactionSystem(unittest.TestCase):
    keq_data = [
        {'name': 'AA', 'data': (-10.743, 3083)},
        {'name': 'BB', 'data': (-10.421, 3166)},
        {'name': 'CC', 'data': (-10.843, 3316)},
        {'name': 'DD', 'data': (-10.136, 3079)}
    ]

    def test_components(self):
        A = Component('A', ComponentType.MONOMER, [('A', 1)])
        AB = Component('AB', ComponentType.DIMER, [('A', 1), ('B', 1)])
        A2 = Component('A2', ComponentType.DIMER, [('A', 2)])
        inert = Component('I', ComponentType.INERT, ['I', 1])

    def test_reaction_system(self):
        system = self._build_example_reaction_system()

    def test_build_true_component(self):
        system = self._build_example_reaction_system()
        monomers = ['A', 'B', 'C', 'D']
        inters = ['I']
        dimers = [monomers[i] + monomers[j]
                  for i in range(len(monomers))
                  for j in range(i, len(monomers))]
        true_components = [*monomers, *inters, *dimers]
        self.assertTrue(sorted(true_components), sorted(system.true_component_names))
        self.assertEqual(len(system.true_component_names), 15)

        expected_element_stoi_matrix = np.array([[1, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                                 [0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 1, 1, 0, 0, 0],
                                                 [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 2, 1, 0],
                                                 [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 2],
                                                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.assertTrue(np.allclose(system._mbg_by_component_matrix, expected_element_stoi_matrix))

    def test_keq_and_mu(self):
        t = 350
        p = 1
        system = self._build_example_reaction_system()
        system.set_keqs(self.keq_data)
        system.set_tp(t, p)
        system.update_keqs(t)
        print(system._keq_per_mmhg_values)
        print(system.mu_by_rt)

    def test_xi(self):
        t = 350
        p = 1
        system = self._build_example_reaction_system()
        system.set_keqs(self.keq_data)

        system._lambdas = np.array([-1] * 5)
        system.set_tp(t, p)
        system._update_xi()
        print(system._xi)

    def test_estimate_lambdas(self):
        t = 360
        p = 2
        system = self._build_example_reaction_system()
        system.set_keqs(self.keq_data)
        system.set_bi_from_zi(np.array([0.2] * 5))
        system.set_tp(t, p)
        lambdas, iters = system.estimate_lambdas_by_fixing_nt(0.75, np.array([-1.0] * 5))
        print(f'estimated lambdas={lambdas} in {iters} iterations')
        self.assertTrue(np.allclose(lambdas, np.array([-2.99465699, -3.53831473, -3.53284552,
                                                       -3.58288624, -1.32175584])))

    def test_solve_lambdas_and_nt(self):
        t = 360
        p = 2
        system = self._build_example_reaction_system()
        system.set_keqs(self.keq_data)
        lambdas_initial = np.array([-1.0] * 5)
        self.solve_equil_system_by_initial_lambdas(system, t, p, lambdas_initial)

        self.assertTrue(np.allclose(system._xi, np.array([0.05437192, 0.03138125, 0.03155478, 0.03000246,
                                                          0.3087816, 0.02974965, 0.06487617, 0.06483558,
                                                          0.06519863, 0.03536947, 0.07069469, 0.07109054,
                                                          0.03532523, 0.07104607, 0.03572195])))
        self.assertAlmostEqual(system._nt, 0.6477069813958342)

    def test_different_lambdas(self):
        t = 360
        p = 2
        system = self._build_example_reaction_system()
        all_nt = []
        all_t = []
        for i in range(10):
            p = 0.5 + i * 0.5
            all_t.append(p)
            system.set_keqs(self.keq_data)
            lambdas_initial = np.array([1.0] * 5)
            nt = self.solve_equil_system_by_initial_lambdas(system, t, p, lambdas_initial)
            all_nt.append(nt)
        plt.plot(all_t, all_nt)
        plt.xlabel('Pressure (atm)')
        plt.ylabel('$n_t$')
        plt.savefig('nt_vs_P.png')

        # self.assertTrue(np.allclose(system._xi, np.array([0.05437192, 0.03138125, 0.03155478, 0.03000246,
        #                                                    0.3087816,  0.02974965, 0.06487617, 0.06483558,
        #                                                    0.06519863, 0.03536947, 0.07069469, 0.07109054,
        #                                                    0.03532523, 0.07104607, 0.03572195])))
        # self.assertAlmostEqual(system._nt, 0.6477069813958342)

    def solve_equil_system_by_initial_lambdas(self, system, t, p, lambdas_initial):
        iters = system.solve(t, p, np.array([0.2] * 5), 0.75, lambdas_initial)
        print(f'Newton solver converged in {iters} iterations')
        composition = {name: float(value) for name, value in zip(system.true_component_names, system._xi)}
        print(f'xi:')
        print(json.dumps(composition, indent=2))
        print(f'nt={system._nt}')
        return system._nt

    def test_q_g_hessian(self):
        t = 350
        p = 2
        nt = 0.75
        system = self._build_example_reaction_system()
        system.set_keqs(self.keq_data)
        system.set_bi_from_zi(np.array([0.2] * 5))
        system.set_tp(t, p)
        lambas = np.array([-1.0] * 5)
        system._lambdas = lambas
        system._update_xi()
        system.set_nt(nt)
        q = system._compute_q()
        g = system._compute_q_gradient()
        h = system._compute_q_hessian()
        print(q)
        print(g)
        num_der = []
        pert = 1e-6
        for i in range(len(lambas)):
            new_labds = lambas.copy()
            new_labds[i] += pert
            system._lambdas = new_labds
            system._update_xi()
            new_q = system._compute_q()
            num_der.append((new_q - q) / pert)
        print(num_der)
        self.assertTrue(np.allclose(g, np.array(num_der)))

        print('\n')
        print(h)
        num_hessian = []
        pert = 1e-6
        for i in range(len(lambas)):
            new_labds = lambas.copy()
            new_labds[i] += pert
            system._lambdas = new_labds
            system._update_xi()
            new_g = system._compute_q_gradient()
            num_hessian.append((new_g - g) / pert)

        print('\n')
        num_hessian = np.array(num_hessian)
        print(num_hessian)
        self.assertTrue(np.allclose(h, num_hessian))

    def test_evaluate_f(self):
        t = 350
        p = 1
        system = self._build_example_reaction_system()
        system.set_keqs(self.keq_data)
        system.set_bi_from_zi(np.array([0.2] * 5))
        system.set_tp(t, p)
        lambdas = np.array([-1.0] * 5)
        system._lambdas = lambdas
        system.set_nt(0.75)
        system._update_xi()
        f = system._compute_f()
        self.assertTrue(np.allclose(f, np.array([12.28901508, 23.32454875, 23.50110656,
                                                 24.32033092, 0.07590958, 56.26030586])))
        jac = system._compute_jac()
        # print(jac)
        pert = 1e-6
        var_size = system.mbg_size + 1
        num_jac = np.zeros((var_size, var_size))
        for i in range(system.mbg_size):
            new_lambdas = lambdas.copy()
            new_lambdas[i] += pert
            system._lambdas = new_lambdas
            system._update_xi()
            new_f = system._compute_f()
            num_der = (new_f - f) / pert
            num_jac[:, i] = num_der

        new_lambdas = lambdas.copy()
        system._lambdas = new_lambdas
        system.set_nt(0.75 + pert)
        system._update_xi()
        new_f = system._compute_f()
        num_der = (new_f - f) / pert
        num_jac[:, -1] = num_der

        # print('\n\n')
        # print(num_jac)
        self.assertTrue(np.allclose(jac, num_jac))

    def _build_example_reaction_system(self):
        monomers = [Component(name, ComponentType.MONOMER, [(name, 1)])
                    for name in ('A', 'B', 'C', 'D')]
        all_components = [*monomers, Component('I', ComponentType.INERT, [('I', 1)])]
        return ReactionSystem(all_components)
