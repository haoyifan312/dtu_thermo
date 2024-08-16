import dataclasses
from enum import IntEnum
from typing import List

import numpy as np


class ComponentType(IntEnum):
    MONOMER = 0
    DIMER = 1
    INERT = 2

@dataclasses.dataclass
class Component:
    name: str
    type: ComponentType
    material_balance_group_and_stoi: List

class ReactionBuilder:
    def __init__(self, app_components: list):
        self.app_components = app_components
        self._mbg = self._extract_material_group()
        self._true_components = self._build_true_components()
        self._mbg_by_component_matrix = self._build_material_group_by_true_components_matrix()

    @property
    def app_component_names(self):
        return [c.name for c in self.app_components]

    @property
    def monomers(self):
        return [c for c in self.app_components if c.type == ComponentType.MONOMER]

    def _build_true_components(self):
        monomers = self.monomers
        monomer_size = len(monomers)
        dimers = []
        for i in range(monomer_size):
            for j in range(i, monomer_size):
                mo_i = monomers[i]
                mo_j = monomers[j]
                mo_i_mbg = mo_i.material_balance_group_and_stoi[0][0]
                mo_j_mbg = mo_j.material_balance_group_and_stoi[0][0]
                dimer_name = mo_i.name + mo_j.name
                if i == j:
                    mbgs = [(mo_i_mbg, 2)]
                else:
                    mbgs = [(mo_i_mbg, 1), (mo_j_mbg, 1)]
                dimers.append(Component(dimer_name, ComponentType.DIMER,
                                        mbgs))
        return [*self.app_components, *dimers]

    @property
    def true_component_names(self):
        return [c.name for c in self._true_components]

    def _build_material_group_by_true_components_matrix(self):
        mbg_map = {mbg: i for i, mbg in enumerate(self._mbg)}
        component_map = {c.name: i for i, c in enumerate(self._true_components)}
        ret = np.zeros((len(self._mbg), len(self._true_components)), dtype=np.uint)
        for component in self._true_components:
            col_index = component_map[component.name]
            for mbg, stoi in component.material_balance_group_and_stoi:
                row_index = mbg_map[mbg]
                ret[row_index, col_index] = stoi
        return ret

    def _extract_material_group(self):
        ret = set()
        for component in self.app_components:
            for mbg_name, _ in component.material_balance_group_and_stoi:
                ret.add(mbg_name)
        return list(ret)


