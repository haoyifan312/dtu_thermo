import dataclasses
from copy import deepcopy

import numpy as np

from RachfordRiceSolver import RachfordRiceBase
from thermclc_interface import ThermclcInterface, FlashInput, PhaseEnum


class SuccessiveSubstitutionException(Exception):
    pass

class SuccessiveSubstitutionSolver:
    def __init__(self, stream: ThermclcInterface, rr=None):
        self._stream = stream
        if rr is None:
            rr = RachfordRiceBase(stream.inflow_size)
        self._rr = rr
        self._max_iter = 1000
        self._ss_tol = 1e-7

    def compute(self, flash_input: FlashInput):
        initial_ks, initial_result = self._calculate_initial_results(flash_input)
        if 0.0 < initial_result.beta < 1.0:
            return self.solve_successive_substitution(flash_input, initial_ks, initial_result)
    def solve_successive_substitution(self, flash_input, initial_ks, initial_result):
        t = flash_input.T
        p = flash_input.P
        last_result = deepcopy(initial_result)
        new_ks = initial_ks.copy()
        for i in range(self._max_iter):
            props_v = self._stream.calc_properties(FlashInput(t, p, last_result.ys), PhaseEnum.VAP)
            props_l = self._stream.calc_properties(FlashInput(t, p, last_result.xs), PhaseEnum.LIQ)
            last_ks = new_ks.copy()
            new_ks = np.exp(props_l.phi) / np.exp(props_v.phi)

            k_diffs = np.abs(new_ks - last_ks)
            if np.max(k_diffs) < self._ss_tol:
                break

            last_result = self._solve_beta_from_rachford_rice(new_ks, flash_input.zs, last_result.beta)
        else:
            raise SuccessiveSubstitutionException(f'Successive substitution solver failed to converge '
                                                  f'at max iterations of {self._max_iter}')
        return i, last_result

    def _calculate_initial_results(self, flash_input: FlashInput):
        t = flash_input.T
        p = flash_input.P
        ks = self._stream.all_wilson_ks(t, p)
        return ks, self._rr.compute(ks, flash_input.zs)

    def _solve_beta_from_rachford_rice(self, new_ks, zs, beta_initial_guess=None):
        return self._rr.compute(new_ks, zs, initial_guess=beta_initial_guess)

