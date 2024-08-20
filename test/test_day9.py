import unittest

import numpy as np

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
        print(system._mbg_by_component_matrix)

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

        system._lambdas = np.array([-1]*5)
        system.set_tp(t, p)
        system._update_xi()
        print(system._xi)

    def test_estimate_lambdas(self):
        t = 360
        p = 2
        system = self._build_example_reaction_system()
        system.set_keqs(self.keq_data)
        system.set_bi_from_zi(np.array([0.2]*5))
        system.set_tp(t, p)
        lambdas, iters = system.estimate_lambdas_by_fixing_nt(0.6, np.array([-1.0]*5))
        print(f'estimated lambdas={lambdas} in {iters} iterations')
        self.assertTrue(np.allclose(lambdas, np.array([-2.86897603, -3.42159276, -3.41605582, -3.46670241,
                                                       -1.09861229])))

    def test_q_g_hessian(self):
        t = 350
        p = 2
        nt = 0.75
        system = self._build_example_reaction_system()
        system.set_keqs(self.keq_data)
        system.set_bi_from_zi(np.array([0.2]*5))
        system.set_tp(t, p)
        lambas = np.array([-1.0]*5)
        system._lambdas = lambas
        system._update_xi()
        q = system._compute_q(nt)
        g = system._compute_q_gradient(nt)
        h = system._compute_q_hessian(nt)
        print(q)
        print(g)
        num_der = []
        pert = 1e-6
        for i in range(len(lambas)):
            new_labds = lambas.copy()
            new_labds[i] += pert
            system._lambdas = new_labds
            system._update_xi()
            new_q = system._compute_q(nt)
            num_der.append((new_q-q)/pert)
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
            new_g = system._compute_q_gradient(nt)
            num_hessian.append((new_g-g)/pert)

        print('\n')
        num_hessian = np.array(num_hessian)
        print(num_hessian)
        self.assertTrue(np.allclose(h, num_hessian))

    def test_evaluate_f(self):
        t = 350
        p = 1
        system = self._build_example_reaction_system()
        system.set_keqs(self.keq_data)
        system.set_bi_from_zi(np.array([0.2]*5))
        system.set_tp(t, p)
        system.estimate_lambdas_by_fixing_nt(0.75, np.array([-1.0]*5))


    def _build_example_reaction_system(self):
        monomers = [Component(name, ComponentType.MONOMER, [(name, 1)])
                    for name in ('A', 'B', 'C', 'D')]
        all_components = [*monomers, Component('I', ComponentType.INERT, [('I', 1)])]
        return ReactionSystem(all_components)
