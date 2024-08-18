import unittest

import numpy as np

from ReactionSystem import ComponentType, Component, ReactionSystem


class TestReactionSystem(unittest.TestCase):
    keq_data = {
        'AA': (-10.743, 3083),
        'BB': (-10.421, 3166),
        'CC': (-10.843, 3316),
        'DD': (-10.136, 3079)
    }

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
        system.update_keqs(t)
        print(system._keq_per_mmhg_values)
        system.update_mu_by_rt(t, p)
        print(system.mu_by_rt)

    def test_xi(self):
        t = 350
        p = 1
        system = self._build_example_reaction_system()
        system.set_keqs(self.keq_data)

        system._lambdas = np.array([-1]*5)
        system._update_xi(t, p)
        print(system._xi)

    def test_estimate_lambdas(self):
        t = 350
        p = 1
        system = self._build_example_reaction_system()
        system.set_keqs(self.keq_data)
        system.set_bi_from_zi(np.array([0.2]*5))
        lambdas, iters = system.estimate_lambdas_by_fixing_nt(t, p, 0.75, np.array([-1.0]*5))
        print(f'estimated lambdas={lambdas} in {iters} iterations')

    def _build_example_reaction_system(self):
        monomers = [Component(name, ComponentType.MONOMER, [(name, 1)])
                    for name in ('A', 'B', 'C', 'D')]
        all_components = [*monomers, Component('I', ComponentType.INERT, [('I', 1)])]
        return ReactionSystem(all_components)
