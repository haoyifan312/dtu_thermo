import unittest

from ReactionBuilder import ComponentType, Component, ReactionBuilder


class TestReactionBuilder(unittest.TestCase):
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

    def _build_example_reaction_system(self):
        monomers = [Component(name, ComponentType.MONOMER, [(name, 1)])
                    for name in ('A', 'B', 'C', 'D')]
        all_components = [*monomers, Component('I', ComponentType.INERT, [('I', 1)])]
        return ReactionBuilder(all_components)
