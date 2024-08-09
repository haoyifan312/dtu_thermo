from copy import deepcopy

import matplotlib.pyplot as plt

from RachfordRiceSolver import RachfordRiceBase, RachfordRiceResult
from SuccessiveSubstitutionSolver import SuccessiveSubstitutionSolver, SuccSubStatus, SSAccelerationDummy
from thermclc_interface import *


class TwoPhaseFlashException(Exception):
    def __init__(self, *args, **kwargs):
        self.result = kwargs.pop('result')
        self.total_rr_count = kwargs.pop('total_rr_count')
        super().__init__(*args, **kwargs)


class TwoPhaseFlash:
    def __init__(self, stream: ThermclcInterface, rr_rigorous=None, rr_fast=None,
                 acceleration=SSAccelerationDummy()):
        self._stream = stream
        if rr_rigorous is None:
            rr_rigorous = RachfordRiceBase(stream.inflow_size)
        if rr_fast is None:
            rr_fast = RachfordRiceBase(stream.inflow_size)
        self._rr_rigous = rr_rigorous
        self._rr_fast = rr_fast
        self._max_iter = 2000
        self._ss_tol = 1e-7
        self._acceleration = acceleration

    def compute(self, flash_input: FlashInput, initial_ks=None, show_plot=False):
        if initial_ks is None:
            initial_ks, initial_result = self._calculate_initial_results_from_wilson_k(flash_input)
        else:
            initial_result = self._rr_rigous.compute(initial_ks, flash_input.zs)
        if initial_result.phase in (PhaseEnum.VAP, PhaseEnum.LIQ):
            if self._stablity_analysis_suggest_single_phase(flash_input, initial_result):
                return 1, initial_result
        return self.solve_successive_substitution(flash_input, initial_ks, initial_result, show_plot=show_plot)

    def solve_successive_substitution(self, flash_input, initial_ks, initial_result, show_plot=False):
        t = flash_input.T
        p = flash_input.P
        last_result = deepcopy(initial_result)
        new_ks = initial_ks.copy()
        beta_history = []
        g_history = []
        rr_counts = [initial_result.rr_iters]

        def compute_ks(last_result, did_acceleration_last_iter):
            beta_history.append(last_result.beta)
            props_v = self._stream.calc_properties(FlashInput(t, p, last_result.ys_or_zs), PhaseEnum.VAP)
            props_l = self._stream.calc_properties(FlashInput(t, p, last_result.xs_or_zs), PhaseEnum.LIQ)
            g_history.append(self.calc_total_G(flash_input, last_result, props_l, props_v))

            g_diff = -1 if len(g_history) < 2 else g_history[-1] - g_history[-2]
            if g_diff < 0.0 or not did_acceleration_last_iter:
                new_ks = np.exp(props_l.phi) / np.exp(props_v.phi)
                return new_ks, SuccSubStatus.CONTINUE, last_result
            elif abs(g_diff) < self._ss_tol * self._ss_tol:
                return None, SuccSubStatus.CONVERGED, last_result
            else:
                return None, SuccSubStatus.OVERSHOOT, last_result

        def converged_fun():
            beta_err_history = [abs(el - beta_history[-1]) for el in beta_history]
            # beta_err_history = [np.log(abs(el)) for el in beta_err_history]
            # print([e for e in beta_err_history])
            if show_plot:
                plt.plot(beta_err_history)
                plt.yscale('log')
                plt.show()
            # print(f'G_history={[float(a) for a in g_history]}')
            # plt.plot(g_history)
            # plt.show()

        def compute_for_next_iter(new_values, last_result):
            last_result = self._solve_beta_from_rachford_rice(new_values, flash_input.zs, last_result.beta)
            rr_counts.append(last_result.rr_iters)
            return last_result

        def max_iter_reached():
            # plt.plot(g_history)
            # plt.show()
            if show_plot:
                beta_err_history = [abs(el - beta_history[-1]) for el in beta_history]
                plt.plot(beta_err_history)
                plt.yscale('log')
                plt.show()
            raise TwoPhaseFlashException(f'Successive substitution solver failed to converge '
                                         f'at max iterations of {self._max_iter}', result=last_result,
                                         total_rr_count=sum(rr_counts))

        self._acceleration.clear()
        ss = SuccessiveSubstitutionSolver(compute_ks, converged_fun, compute_for_next_iter,
                                          max_iter_reached, acceleration=self._acceleration)
        ss_iters, last_result = ss.solve(new_ks, last_result)

        return ss_iters, sum(rr_counts), last_result

    def _calculate_initial_results_from_wilson_k(self, flash_input: FlashInput):
        t = flash_input.T
        p = flash_input.P
        ks = self._stream.all_wilson_ks(t, p)
        return ks, self._rr_rigous.compute(ks, flash_input.zs)

    def _solve_beta_from_rachford_rice(self, new_ks, zs, beta_initial_guess=None):
        if self._acceleration.did:
            return self._rr_rigous.compute(new_ks, zs, initial_guess=beta_initial_guess)
        else:
            return self._rr_fast.compute(new_ks, zs, initial_guess=beta_initial_guess)

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
            raise TwoPhaseFlashException(f'Initial VLE guess should not be for beta={beta}')
        effective_ks = ret.ys / ret.xs
        return effective_ks, ret

    def _stablity_analysis_suggest_single_phase(self, flash_input, initial_result):
        initial_phase = initial_result.phase
        if initial_phase not in (PhaseEnum.VAP, PhaseEnum.LIQ):
            raise TwoPhaseFlashException(f'Phase {initial_result.phase.name} '
                                         f'should not perform stability analysis')



        ks = self._stream.all_wilson_ks(flash_input.T, flash_input.P)
        # new_ws = estimate_heavy_phase_from_wilson_ks(flash_input.zs, ks) if initial_result.phase == PhaseEnum.VAP else \
        #     estimate_light_phase_from_wilson_ks(flash_input.zs, ks)


    def calc_total_G(self, flash_input, result, props_l, props_v):
        total_z = np.sum(flash_input.zs)
        total_v = total_z * result.beta
        total_l = total_z - total_v
        vi = total_v * result.ys
        li = total_l * result.xs
        log_yi = np.log(result.ys) if total_v > 0.0 else np.zeros(self._stream.inflow_size)
        log_xi = np.log(result.xs) if total_l > 0.0 else np.zeros(self._stream.inflow_size)
        return np.sum(vi * (log_yi + props_v.phi)) + np.sum(li * (log_xi + props_l.phi))