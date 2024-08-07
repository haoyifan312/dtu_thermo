from copy import deepcopy

import numpy as np

from RachfordRiceSolver import RachfordRiceBase, RachfordRiceResult
from thermclc_interface import *


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

    def compute(self, flash_input: FlashInput, initial_ks=None):
        if initial_ks is None:
            initial_ks, initial_result = self._calculate_initial_results_from_wilson_k(flash_input)
        else:
            initial_result = self._rr.compute(initial_ks, flash_input.zs)
        if initial_result.phase in (PhaseEnum.VAP, PhaseEnum.LIQ):
            if self._stablity_analysis_suggest_single_phase(flash_input, initial_result):
                return 1, initial_result
        return self.solve_successive_substitution(flash_input, initial_ks, initial_result)

    def solve_successive_substitution(self, flash_input, initial_ks, initial_result):
        t = flash_input.T
        p = flash_input.P
        last_result = deepcopy(initial_result)
        new_ks = initial_ks.copy()
        for i in range(self._max_iter):
            props_v = self._stream.calc_properties(FlashInput(t, p, last_result.ys_or_zs), PhaseEnum.VAP)
            props_l = self._stream.calc_properties(FlashInput(t, p, last_result.xs_or_zs), PhaseEnum.LIQ)
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

    def _calculate_initial_results_from_wilson_k(self, flash_input: FlashInput):
        t = flash_input.T
        p = flash_input.P
        ks = self._stream.all_wilson_ks(t, p)
        return ks, self._rr.compute(ks, flash_input.zs)

    def _solve_beta_from_rachford_rice(self, new_ks, zs, beta_initial_guess=None):
        return self._rr.compute(new_ks, zs, initial_guess=beta_initial_guess)

    def _get_initial_ks_without_material_balance(self, flash_input, beta):
        ret = RachfordRiceResult(beta=None)
        wilson_ks = np.array(self._stream.all_wilson_ks(flash_input.T, flash_input.P))
        if beta == 1.0:
            ret.xs = estimate_heavy_phase_from_wilson_ks(flash_input.zs, wilson_ks)
            ret.normalize_xs()
            ret.ys = flash_input.zs.copy()
        elif beta == 0.0:
            ret.ys = estimate_light_phase_from_wilson_ks(flash_input.zs, wilson_ks)
            ret.normalize_ys()
            ret.xs = flash_input.zs.copy()
        else:
            raise SuccessiveSubstitutionException(f'Initial VLE guess should not be for beta={beta}')
        effective_ks = ret.ys/ret.xs
        return effective_ks, ret

    def _stablity_analysis_suggest_single_phase(self, flash_input, initial_result):
        initial_phase = initial_result.phase
        if initial_phase not in (PhaseEnum.VAP, PhaseEnum.LIQ):
            raise SuccessiveSubstitutionException(f'Phase {initial_result.phase.name} '
                                                  f'should not perform stability analysis')

        ln_phi_z = self._stream.calc_properties(flash_input, initial_phase).phi
        di = ln_phi_z + np.log(flash_input.zs)

        ks = self._stream.all_wilson_ks(flash_input.T, flash_input.P)
        # new_ws = estimate_heavy_phase_from_wilson_ks(flash_input.zs, ks) if initial_result.phase == PhaseEnum.VAP else \
        #     estimate_light_phase_from_wilson_ks(flash_input.zs, ks)
        new_ws = estimate_heavy_phase_from_wilson_ks_aggressive(flash_input.zs, ks, ln_phi_z)
        new_ln_ws = np.log(new_ws)
        new_flash_input = deepcopy(flash_input)
        for i in range(self._max_iter):
            new_flash_input.zs = new_ws.copy()
            new_w_props = self._stream.calc_properties(new_flash_input, PhaseEnum.STABLE)
            bracket_term = new_ln_ws + new_w_props.phi - di - 1.0
            tm = 1.0 + np.sum(new_ws*bracket_term)

            # check tm < 0.0

            old_ln_ws = new_ln_ws.copy()
            new_ln_ws = di - new_w_props.phi
            new_ws = np.exp(new_ln_ws)

            diff = np.abs(new_ln_ws - old_ln_ws)
            if np.max(diff) < self._ss_tol:
                return True
        else:
            raise SuccessiveSubstitutionException(f'Successive substitution on tm reached '
                                                  f'max iterations {self._max_iter}')










