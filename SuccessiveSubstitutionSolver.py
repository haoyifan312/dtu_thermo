from copy import deepcopy

import matplotlib.pyplot as plt

from RachfordRiceSolver import RachfordRiceBase, RachfordRiceResult
from thermclc_interface import *


class SuccessiveSubstitutionException(Exception):
    def __init__(self, *args, **kwargs):
        self.result = kwargs.pop('result')
        super().__init__(*args, **kwargs)


class SSAccelerationDummy:
    def check(self, current_iter, ss_var_history, new_var):
        return new_var

    @property
    def did(self):
        """dummy accelerator never accelerate"""
        return False

    @property
    def counter(self):
        return 0

class SSAccelerationCycle:
    def __init__(self, cycle=5):
        self._cycle = cycle
        self._did = False
        self._counter = 0

    def check(self, current_iter, ss_var_history, new_var):
        self._did = False
        if current_iter == 0 or current_iter % self._cycle > 0:
            return new_var
        y_kp2 = new_var
        y_kp1 = ss_var_history[-1]
        y_k = ss_var_history[-2]
        d_k = y_kp1 - y_k
        d_kp1 = y_kp2 - y_kp1
        lam_denom = np.sum(d_k * d_kp1)
        if abs(lam_denom) < 1e-30:
            return new_var
        lam = np.sum(d_kp1) ** 2 / lam_denom
        y_inf_denom = 1.0 - lam
        if abs(y_inf_denom) < 1e-10:
            return new_var
        y_inf = y_kp2 + d_kp1 * lam /y_inf_denom

        self._did = True
        self._counter += 1
        return y_inf

    @property
    def did(self):
        return self._did

    @property
    def counter(self):
        return self._counter


class SuccessiveSubstitutionSolver:
    def __init__(self, stream: ThermclcInterface, rr=None, acceleration=SSAccelerationDummy()):
        self._stream = stream
        if rr is None:
            rr = RachfordRiceBase(stream.inflow_size)
        self._rr = rr
        self._max_iter = 1000
        self._ss_tol = 1e-7
        self._acceleration = acceleration

    def compute(self, flash_input: FlashInput, initial_ks=None, show_plot=False):
        if initial_ks is None:
            initial_ks, initial_result = self._calculate_initial_results_from_wilson_k(flash_input)
        else:
            initial_result = self._rr.compute(initial_ks, flash_input.zs)
        if initial_result.phase in (PhaseEnum.VAP, PhaseEnum.LIQ):
            if self._stablity_analysis_suggest_single_phase(flash_input, initial_result):
                return 1, initial_result
        return self.solve_successive_substitution(flash_input, initial_ks, initial_result, show_plot=show_plot)

    def solve_successive_substitution(self, flash_input, initial_ks, initial_result, show_plot=False):
        t = flash_input.T
        p = flash_input.P
        last_result = deepcopy(initial_result)
        new_ks = initial_ks.copy()
        ks_history = []
        beta_history = []
        g_history = []
        g_diff = -1
        for i in range(self._max_iter):
            beta_history.append(last_result.beta)
            props_v = self._stream.calc_properties(FlashInput(t, p, last_result.ys_or_zs), PhaseEnum.VAP)
            props_l = self._stream.calc_properties(FlashInput(t, p, last_result.xs_or_zs), PhaseEnum.LIQ)
            g_history.append(self._calc_total_G(flash_input, last_result, props_l, props_v))
            if i > 0:
                g_diff = g_history[-1] - g_history[-2]
            last_ks = new_ks.copy()
            ks_history.append(last_ks)

            if g_diff < 0.0 or not self._acceleration.did:
                new_ks = np.exp(props_l.phi) / np.exp(props_v.phi)
                new_ks = self._acceleration.check(i, ks_history, new_ks)
            else:
                new_ks = ks_history[-2]

            k_diffs = np.abs(new_ks - last_ks)
            if np.max(k_diffs) < self._ss_tol:
                beta_err_history = [abs(el - beta_history[-1]) for el in beta_history]
                # beta_err_history = [np.log(abs(el)) for el in beta_err_history]
                print([e for e in beta_err_history])
                if show_plot:
                    plt.plot(beta_err_history)
                    plt.yscale('log')
                    plt.show()
                print(f'G_history={[float(a) for a in g_history]}')
                # plt.plot(g_history)
                # plt.show()
                break

            last_result = self._solve_beta_from_rachford_rice(new_ks, flash_input.zs, last_result.beta)
        else:
            # plt.plot(g_history)
            # plt.show()
            if show_plot:
                beta_err_history = [abs(el - beta_history[-1]) for el in beta_history]
                plt.plot(beta_err_history)
                plt.yscale('log')
                plt.show()
            raise SuccessiveSubstitutionException(f'Successive substitution solver failed to converge '
                                                  f'at max iterations of {self._max_iter}', result=last_result)
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
        effective_ks = ret.ys / ret.xs
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
            tm = 1.0 + np.sum(new_ws * bracket_term)

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

    def _calc_total_G(self, flash_input, result, props_l, props_v):
        total_z = np.sum(flash_input.zs)
        total_v = total_z * result.beta
        total_l = total_z - total_v
        vi = total_v * result.ys
        li = total_l * result.xs
        log_yi = np.log(result.ys) if total_v > 0.0 else np.zeros(self._stream.inflow_size)
        log_xi = np.log(result.xs) if total_l > 0.0 else np.zeros(self._stream.inflow_size)
        return np.sum(vi * (log_yi + props_v.phi)) + np.sum(li * (log_xi + props_l.phi))
