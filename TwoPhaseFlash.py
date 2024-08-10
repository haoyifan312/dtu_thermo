from copy import deepcopy

import matplotlib.pyplot as plt

from RachfordRiceSolver import RachfordRiceBase, RachfordRiceResult
from StabilityAnalysis import StabilityAnalysis, SAResultType
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
        self._ss_max_iter = 2000
        self._ss_tol = 1e-7
        self._acceleration = acceleration
        self._sa = StabilityAnalysis(self._stream, ss_max_iter=self._ss_max_iter, ss_tol=self._ss_tol)

    def compute(self, flash_input: FlashInput, initial_ks=None, show_plot=False):
        if initial_ks is None:
            initial_ks, initial_result = self._calculate_initial_results_from_wilson_k(flash_input)
        else:
            initial_result = self._rr_rigous.compute(initial_ks, flash_input.zs)
        if initial_result.phase in (PhaseEnum.VAP, PhaseEnum.LIQ):
            if self._stablity_analysis_suggest_single_phase(flash_input, initial_result):
                return 1, initial_result
        return self.solve_successive_substitution(flash_input, initial_ks, initial_result, show_plot=show_plot)

    def solve_successive_substitution(self, flash_input, initial_ks, initial_result: RachfordRiceResult,
                                      show_plot=False, checked_stability_analysis=False):
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
                                         f'at max iterations of {self._ss_max_iter}', result=last_result,
                                         total_rr_count=sum(rr_counts))

        self._acceleration.clear()
        ss = SuccessiveSubstitutionSolver(compute_ks, converged_fun, compute_for_next_iter,
                                          max_iter_reached, acceleration=self._acceleration,
                                          max_iter=self._ss_max_iter, tol=self._ss_tol)
        ss_iters, last_result = ss.solve(new_ks, last_result)

        if not checked_stability_analysis and last_result.phase != PhaseEnum.VLE and self._need_stability_analysis():
            is_two_phase, sa_ks = self._stability_analysis_suggest_two_phase(flash_input, last_result)
            if is_two_phase:
                sa_result = self._rr_rigous.compute(sa_ks, flash_input.zs)
                return self.solve_successive_substitution(flash_input, sa_ks, sa_result, show_plot=show_plot,
                                                          checked_stability_analysis=True)

        self._flip_phase_if_reversed(flash_input, last_result)

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

    def _stability_analysis_suggest_two_phase(self, flash_input, last_result: RachfordRiceResult):
        wilson_ks = self._stream.all_wilson_ks(flash_input.T, flash_input.P)
        if last_result.phase == PhaseEnum.LIQ:
            wi_guess = estimate_light_phase_from_wilson_ks(flash_input.zs, wilson_ks)
            z_prop = self._stream.calc_properties(flash_input, PhaseEnum.LIQ)
            ln_phi_z = z_prop.phi
        else:
            wi_guess = estimate_heavy_phase_from_wilson_ks(flash_input.zs, wilson_ks)
            z_prop = self._stream.calc_properties(flash_input, PhaseEnum.VAP)
            ln_phi_z = z_prop.phi
        sa_result, _ = self._sa.compute(flash_input, wi_guess)
        if sa_result.category == SAResultType.NEGATIVE:
            mole_frac = sa_result.wi/np.sum(sa_result.wi)
            wi_flash_input = FlashInput(flash_input.T, flash_input.P, mole_frac)
            if last_result.phase == PhaseEnum.LIQ:
                x_prop = self._stream.calc_properties(wi_flash_input, PhaseEnum.LIQ)
                ln_phi_x = x_prop.phi
                ks = np.exp(ln_phi_x)/np.exp(ln_phi_z)
            else:
                y_prop = self._stream.calc_properties(wi_flash_input, PhaseEnum.VAP)
                ln_phi_y = y_prop.phi
                ks = np.exp(ln_phi_z)/np.exp(ln_phi_y)
            return True, ks
        return False, None

    def _flip_phase_if_reversed(self, flash_input: FlashInput, last_result: RachfordRiceResult):
        if last_result.phase != PhaseEnum.VLE:
            return
        input_x = FlashInput(flash_input.T, flash_input.P, last_result.xs)
        prop_x = self._stream.calc_properties(input_x, desired_phase=PhaseEnum.STABLE)
        input_y = FlashInput(flash_input.T, flash_input.P, last_result.ys)
        prop_y = self._stream.calc_properties(input_y, desired_phase=PhaseEnum.STABLE)
        if prop_x.phase == PhaseEnum.VAP and prop_y.phase == PhaseEnum.LIQ:
            save_xs = last_result.xs.copy()
            last_result.xs = last_result.ys.copy()
            last_result.ys = save_xs
            last_result.beta = 1.0-last_result.beta

    def _need_stability_analysis(self):
        return self._rr_rigous.is_stability_analysis_needed() or self._rr_rigous.is_stability_analysis_needed()





