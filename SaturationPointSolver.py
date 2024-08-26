from enum import IntEnum
import numpy as np
import matplotlib.pyplot as plt

from RachfordRiceSolver import RachfordRiceBase, RachfordRiceSolverOption, RachfordRiceResult
from thermclc_interface import ThermclcInterface, PropertyType, FlashInput, PhaseEnum


class SaturationPointException(Exception):
    def __init__(self, *args, **kwargs):
        self.tp_ki = kwargs.get('tp_ki')
        try:
            kwargs.pop('tp_ki')
        except KeyError:
            pass
        super().__init__(*args, **kwargs)


class SaturationType(IntEnum):
    BUBBLE_POINT = 1
    DEW_POINT = 2

    @staticmethod
    def from_beta(beta):
        if beta == 1.0:
            return SaturationType.DEW_POINT
        elif beta == 0.0:
            return SaturationType.BUBBLE_POINT


class BubblePointUtilities:
    def fun(self, zi: np.array, ki: np.array):
        return np.sum(zi * ki) - 1.0

    def der(self, zi: np.array, ki: np.array, dlnK_der: np.array):
        return np.sum(zi * ki * dlnK_der)

    def set_result(self, rr_result: RachfordRiceResult, zi, ki):
        rr_result.ys = zi * ki

    def ln_k_props_from_ln_phi_diff(self, stream: ThermclcInterface, t, p, zi: np.array,
                                    rr_result: RachfordRiceResult):
        yi = rr_result.ys_or_zs
        ln_phi_l_props = stream.calc_properties(FlashInput(t, p, zi), PhaseEnum.LIQ)
        ln_phi_v_props = stream.calc_properties(FlashInput(t, p, yi), PhaseEnum.VAP)
        return ln_phi_l_props - ln_phi_v_props, ln_phi_l_props.phase == ln_phi_v_props.phase

    def leads_to_trivial_solution(self, stream: ThermclcInterface, t, p, zi: np.array,
                                  rr_result: RachfordRiceResult):
        yi = rr_result.ys_or_zs
        ln_phi_l_props = stream.calc_properties(FlashInput(t, p, zi), PhaseEnum.LIQ)
        ln_phi_v_props = stream.calc_properties(FlashInput(t, p, yi), PhaseEnum.VAP)
        return ln_phi_l_props.phase == ln_phi_v_props.phase


class DewPointUtilities:
    def fun(self, zi: np.array, ki: np.array):
        return np.sum(zi / ki) - 1.0

    def der(self, zi: np.array, ki: np.array, dlnK_der: np.array):
        return - np.sum(zi / ki * dlnK_der)

    def set_result(self, rr_result: RachfordRiceResult, zi, ki):
        rr_result.xs = zi / ki

    def ln_k_props_from_ln_phi_diff(self, stream: ThermclcInterface, t, p, zi: np.array,
                                    rr_result: RachfordRiceResult):
        xi = rr_result.xs_or_zs
        ln_phi_l_props = stream.calc_properties(FlashInput(t, p, xi), PhaseEnum.LIQ)
        ln_phi_v_props = stream.calc_properties(FlashInput(t, p, zi), PhaseEnum.VAP)
        return ln_phi_l_props - ln_phi_v_props, ln_phi_l_props.phase == ln_phi_v_props.phase

    def leads_to_trivial_solution(self, stream: ThermclcInterface, t, p, zi: np.array, rr_result: RachfordRiceResult):
        xi = rr_result.xs_or_zs
        ln_phi_l_props = stream.calc_properties(FlashInput(t, p, xi), PhaseEnum.LIQ)
        ln_phi_v_props = stream.calc_properties(FlashInput(t, p, zi), PhaseEnum.VAP)
        return ln_phi_l_props.phase == ln_phi_v_props.phase


class BubblePoint:
    def __init__(self):
        self._utilities = BubblePointUtilities()

    def fun(self, zi: np.array, ki: np.array):
        return self._utilities.fun(zi, ki)

    def der(self, zi: np.array, ki: np.array, k_ders: np.array):
        return np.sum(zi * k_ders)


class BubblePointForPhi(BubblePoint):
    def compute_ln_ki_props(self, t, p, zi, yi, stream: ThermclcInterface):
        prop_l_z = stream.calc_properties(FlashInput(t, p, zi), PhaseEnum.LIQ)
        prop_v_y = stream.calc_properties(FlashInput(t, p, yi), PhaseEnum.VAP)
        return prop_l_z - prop_v_y


class DewPoint:
    def __init__(self):
        self._utilities = DewPointUtilities()

    def fun(self, zi: np.array, ki: np.array):
        return self._utilities.fun(zi, ki)

    def der(self, zi: np.array, ki: np.array, k_ders: np.array):
        return - np.sum(zi / ki / ki * k_ders)


class DewPointForPhi(DewPoint):
    def compute_ln_ki_props(self, t, p, zi, xi, stream: ThermclcInterface):
        prop_v_z = stream.calc_properties(FlashInput(t, p, zi), PhaseEnum.VAP)
        prop_l_x = stream.calc_properties(FlashInput(t, p, xi), PhaseEnum.LIQ)
        return prop_l_x - prop_v_z


class SaturationPointSolver:
    def __init__(self, stream: ThermclcInterface, eqns, max_iter=1000, tol=1e-7):
        self._stream = stream
        self._eqns = eqns
        self._max_iter = max_iter
        self._tol = tol

    def calculate_saturation_condition(self, zi, t, p, free_var: str, damping_factor=1.0):
        if free_var == 'T':
            der_type = PropertyType.TEMPERATURE_DER
            free_var_index = 0
        elif free_var == 'P':
            der_type = PropertyType.PRESSURE_DER
            free_var_index = 1
        else:
            raise SaturationPointException('Either temperature or pressure need to be freed '
                                           'for saturation point calculation')
        tp = [t, p]
        for i in range(self._max_iter):
            ki = self._compute_ki(i, *tp, PropertyType.PROPERTY)
            f = self._eqns.fun(zi, ki)
            if abs(f) < self._tol:
                break

            k_der = self._compute_ki(i, *tp, der_type)
            f_prime = self._eqns.der(zi, ki, k_der)
            if abs(f_prime) < 1e-50:
                raise SaturationPointException(f'Newton derivative is {f_prime}')

            newton_step = -f / f_prime
            if abs(newton_step) < self._tol * self._tol:
                raise SaturationPointException(f'Newton step is reduced to {newton_step} and cannot converge')

            tp[free_var_index] += newton_step * damping_factor
        else:
            raise SaturationPointException(f'Newton solver cannot converge in {i} iterations')
        return tp, ki, i

    def _compute_ki(self, i, t, p, property_type: PropertyType):
        raise NotImplementedError()


class SaturationPointSolverWilson(SaturationPointSolver):

    def _compute_ki(self, i, t, p, property_type: PropertyType):
        return self._stream.all_wilson_ks(t, p, property_type=property_type)


class SaturationPointSolverPhi(SaturationPointSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zi = kwargs.get('zi')
        self._incipient_phase_xi = None

        stream = args[0]
        self._rr = RachfordRiceBase.create_solver(stream.inflow_size, RachfordRiceSolverOption.BASE)

    def set_zi(self, zi: np.array):
        self._zi = zi

    def set_incipient_x(self, xi: np.array):
        self._incipient_phase_xi = xi

    def _compute_ki(self, i, t, p, property_type: PropertyType):
        diff = self._eqns.compute_ln_ki_props(t, p, self._zi, self._incipient_phase_xi, self._stream)
        k = np.exp(diff.phi)
        if property_type == PropertyType.PROPERTY:
            return k
        elif property_type == PropertyType.TEMPERATURE_DER:
            return k * diff.dphi_dt
        elif property_type == PropertyType.PRESSURE_DER:
            return k * diff.dphi_dp


def create_saturation_point_solver(stream: ThermclcInterface, flash_type: SaturationType, solver_name: str):
    gov_eqn_wilson = {
        SaturationType.DEW_POINT: DewPoint(),
        SaturationType.BUBBLE_POINT: BubblePoint()
    }

    gov_eqn_phi = {
        SaturationType.DEW_POINT: DewPointForPhi(),
        SaturationType.BUBBLE_POINT: BubblePointForPhi()
    }

    solvers = {
        'Wilson': (SaturationPointSolverWilson, gov_eqn_wilson),
        'Phi': (SaturationPointSolverPhi, gov_eqn_phi)
    }

    solver, eqns = solvers[solver_name]
    return solver(stream, eqns[flash_type])


class SaturationPointBySuccessiveSubstitution:
    def __init__(self, stream: ThermclcInterface, utilities, initialization_solver,
                 max_iter=1000, tol=1e-7):
        self._stream = stream
        self._utilities = utilities
        self._initialization_solver = initialization_solver
        self._rr = RachfordRiceBase.create_solver(stream.inflow_size, RachfordRiceSolverOption.BASE)
        self._max_iter = max_iter
        self._tol = tol

    def solve(self, t, p, zi: np.array, free_var: str, initialization_data=None, plot_t_vs_k6=None):
        if initialization_data is None:
            initial_tp, initial_ks, _ = self._initialization_solver.calculate_saturation_condition(zi, t, p, free_var)
        else:
            initial_tp, initial_ks = initialization_data
        rr_result = self._rr.compute(initial_ks, zi)
        tp = initial_tp
        if free_var == 'T':
            free_var_index = 0
        elif free_var == 'P':
            free_var_index = 1

        free_var_history = []
        k_history = []
        newton_step = 1.0
        damping_factor = 1.0
        for i in range(self._max_iter):
            free_var_history.append(tp[free_var_index])
            ln_k_props, is_trivial = self._utilities.ln_k_props_from_ln_phi_diff(self._stream, *tp, zi, rr_result)
            ln_k = ln_k_props.phi  # lnK = ln_phi_l - ln_phi_v
            ki = np.exp(ln_k)
            k_history.append(ki)
            f = self._utilities.fun(zi, ki)
            if abs(f) < self._tol and abs(newton_step) < 1e-4:
                if plot_t_vs_k6 is not None:
                    self._plot_t_vs_k6(free_var_history, k_history, plot_t_vs_k6)
                return tp, ki, i
            if free_var == 'T':
                ln_phi_der = ln_k_props.dphi_dt
            elif free_var == 'P':
                ln_phi_der = ln_k_props.dphi_dp
            f_der = self._utilities.der(zi, ki, ln_phi_der)
            if abs(f_der) < 1e-7:
                return tp, ki, i
            newton_step = - f / f_der
            newton_step = self._get_newton_step_to_avoid_bounding(tp, free_var_index, newton_step, zi, rr_result,
                                                                  is_trivial)
            tp[free_var_index] += newton_step * damping_factor
            self._utilities.set_result(rr_result, zi, ki)
        else:
            raise SaturationPointException(f'Saturation point did not converge in {i} iterations', tp_ki=(tp, ki))

    @staticmethod
    def create_saturation_pt_by_successive_substitution(stream: ThermclcInterface, flash_type: SaturationType,
                                                        max_iter=1000):
        if flash_type == SaturationType.BUBBLE_POINT:
            return SaturationPointBySuccessiveSubstitution(stream, BubblePointUtilities(),
                                                           create_saturation_point_solver(stream, flash_type,
                                                                                          'Wilson'),
                                                           max_iter=max_iter)
        elif flash_type == SaturationType.DEW_POINT:
            return SaturationPointBySuccessiveSubstitution(stream, DewPointUtilities(),
                                                           create_saturation_point_solver(stream, flash_type,
                                                                                          'Wilson'),
                                                           max_iter=max_iter)

    def _plot_t_vs_k6(self, t_history, k_history, plot_file_name):
        t_diff_history = [abs(t - t_history[-1]) for t in t_history]
        t_diff_history.pop(-1)
        k_diff_history = [abs(k[6] - k_history[-1][6]) for k in k_history]
        k_diff_history.pop(-1)

        x = [i for i, _ in enumerate(t_diff_history)]
        plt.plot(x, t_diff_history, label='T')
        plt.plot(x, k_diff_history, label='k6')
        plt.legend()
        plt.yscale('log')
        plt.savefig(plot_file_name)

    def _get_newton_step_to_avoid_bounding(self, tp, free_var_index, newton_step, zi, rr_result, was_trivial):
        factor = 1.0
        while True:
            tp_copy = tp.copy()
            effective_newton_step = newton_step * factor
            tp_copy[free_var_index] += effective_newton_step
            lead_to_trivial = not was_trivial
            if not was_trivial:
                lead_to_trivial = self._utilities.leads_to_trivial_solution(self._stream, *tp_copy, zi,
                                                                            rr_result)
            if tp_copy[free_var_index] > 0.0 and not lead_to_trivial:
                return effective_newton_step
            if factor < 1e-5:
                return effective_newton_step
            factor *= 0.5
