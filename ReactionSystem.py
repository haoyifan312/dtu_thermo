import dataclasses
from enum import IntEnum
from typing import List, Any

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


class ReactionSystem:
    _true_components: list[Any]

    def __init__(self, app_components: list, max_iter=100, tol=1e-7):
        self.app_components = app_components
        self._mbg = self._extract_material_group()
        self._true_components = self._build_true_components()
        self._mbg_by_component_matrix = self._build_material_group_by_true_components_matrix()
        # print(self._mbg_by_component_matrix)
        self.keq_data = None
        self.mu_by_rt = np.zeros(len(self._true_components))
        self._keq_index = self._create_keq_index()
        self._keq_per_mmhg_values = np.zeros(len(self.dimers))
        self._lambdas = np.zeros(len(app_components))
        self._bi = np.zeros(len(app_components))
        self._xi = np.zeros(len(self._true_components))
        self._max_iter = max_iter
        self._tol = tol
        self._t = 0.0
        self._p = 0.0

    def set_tp(self, t, p):
        self._t = t
        self._p = p
        self.update_mu_by_rt()

    def set_bi_from_zi(self, zi):
        self._bi = zi

    def _update_xi(self):
        first_term = np.matmul(np.transpose(self._mbg_by_component_matrix), np.transpose(self._lambdas))
        ln_xi = np.transpose(first_term) - self.mu_by_rt
        self._xi = np.exp(ln_xi)

    def set_keqs(self, keqs: dict):
        self.keq_data = keqs

    def update_keqs(self, t):
        dimer_keqs = {keq['name']: self._compute_keq(keq['data'], t)
                                for keq in self.keq_data}
        self_dimers = [keq['name'] for keq in self.keq_data]
        # print(self_dimers)
        self_dimer_size = len(self_dimers)
        for i in range(self_dimer_size):
            for j in range(i+1, self_dimer_size):
                self_dimer_i = self_dimers[i]
                self_dimer_j = self_dimers[j]
                cross_dimer_name = self_dimer_i[0] + self_dimer_j[0]
                dimer_keqs[cross_dimer_name] = 2.0*np.sqrt(dimer_keqs[self_dimer_i]*dimer_keqs[self_dimer_j])

        for dimer, keq in dimer_keqs.items():
            self._keq_per_mmhg_values[self._keq_index[dimer]] = keq

    def _compute_keq(self, data, t):
        k1, k2 = data
        return np.pow(10.0, k1 + k2/t)

    def update_mu_by_rt(self):
        self.update_keqs(self._t)
        p_mmhg = self._p * 760
        self.mu_by_rt = np.zeros(len(self._true_components))
        for i, component in enumerate(self._true_components):
            if component.type != ComponentType.DIMER:
                continue
            keq_index = self._keq_index[component.name]
            keq = self._keq_per_mmhg_values[keq_index]
            self.mu_by_rt[i] = -np.log(keq*p_mmhg)

    @property
    def app_component_names(self):
        return [c.name for c in self.app_components]

    @property
    def monomers(self):
        return [c for c in self.app_components if c.type == ComponentType.MONOMER]

    @property
    def dimers(self):
        return [c for c in self._true_components if c.type == ComponentType.DIMER]

    @property
    def dimer_names(self):
        return [c.name for c in self._true_components if c.type == ComponentType.DIMER]

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
        return sorted(list(ret))

    def _create_keq_index(self):
        return {name: i for i, name in enumerate(self.dimer_names)}

    def estimate_lambdas_by_fixing_nt(self,nt, lambdas):
        self._lambdas = lambdas
        for i in range(self._max_iter):
            # print(lambdas)
            self._update_xi()
            q = self._compute_q(nt)
            g = self._compute_q_gradient(nt)
            # print(g)
            h = self._compute_q_hessian(nt)
            newton_step = np.matmul(np.linalg.inv(h), -g)
            damping_factor = self._compute_damping_factor_to_reduce_q(nt, q, newton_step)
            if np.linalg.norm(damping_factor*newton_step) < 1e-12:
                return lambdas, i
            lambdas += newton_step*damping_factor
            self._lambdas = lambdas


    def _compute_q(self, nt):
        sum_x = np.sum(self._xi)
        sec_term = np.sum(self._lambdas*self._bi)
        return nt*(sum_x - 1.0) - sec_term

    def _compute_q_gradient(self, nt):
        sum_Aji_xi = np.matmul(self._mbg_by_component_matrix, np.transpose(self._xi))
        return nt*np.transpose(sum_Aji_xi) - self._bi

    def _compute_q_hessian(self, nt):
        mbg_size = len(self.app_components)
        ret = np.zeros((mbg_size, mbg_size))
        for j in range(mbg_size):
            for k in range(mbg_size):
                ret[j, k] = nt*np.sum(self._mbg_by_component_matrix[j, :]*self._mbg_by_component_matrix[k, :]*self._xi)
        return ret

    def _compute_damping_factor_to_reduce_q(self, nt, q, newton_step):
        delta = 1e-10
        factor = 1.0
        current_lambdas = self._lambdas.copy()
        while True:
            new_lambdas = current_lambdas + newton_step*factor
            self._lambdas = new_lambdas
            self._update_xi()
            new_q = self._compute_q(nt)
            if new_q < q + delta or factor < 1e-10:
                return factor
            factor *= 0.5



