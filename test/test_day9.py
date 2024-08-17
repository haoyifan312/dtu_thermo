import unittest

import numpy as np

from ReactionBuilder import ComponentType, Component, ReactionBuilder


class TestReactionBuilder(unittest.TestCase):
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

    def test_reaction_builder(self):
        builder = self._build_example_reaction_system()

    def test_build_true_component(self):
        builder = self._build_example_reaction_system()
        monomers = ['A', 'B', 'C', 'D']
        inters = ['I']
        dimers = [monomers[i] + monomers[j]
                    for i in range(len(monomers))
                    for j in range(i, len(monomers))]
        true_components = [*monomers, *inters, *dimers]
        self.assertTrue(sorted(true_components), sorted(builder.true_component_names))
        self.assertEqual(len(builder.true_component_names), 15)
        print(builder._mbg_by_component_matrix)

    def test_keq_and_mu(self):
        t = 350
        p = 1
        builder = self._build_example_reaction_system()
        builder.set_keqs(self.keq_data)
        builder.update_keqs(t)
        print(builder._keq_per_mmhg_values)
        builder.update_mu_by_rt(t, p)
        print(builder.mu_by_rt)

    def test_ln_xi(self):
        t = 350
        p = 1
        builder = self._build_example_reaction_system()
        builder.set_keqs(self.keq_data)
        builder.update_mu_by_rt(t, p)

        builder.set_bi_from_zi(np.array([0.2]*5))
        builder._lambdas = np.array([-1]*5)
        builder._update_xi()
        print(builder._xi)

    def _build_example_reaction_system(self):
        monomers = [Component(name, ComponentType.MONOMER, [(name, 1)])
                    for name in ('A', 'B', 'C', 'D')]
        all_components = [*monomers, Component('I', ComponentType.INERT, [('I', 1)])]
        return ReactionBuilder(all_components)
