import numpy as np

from SaturationPointSolver import SaturationType, create_saturation_point_solver
from thermclc_interface import ThermclcInterface, FlashInput, PhaseEnum


class EquilEqnsForSaturationPointException(Exception):
    pass


class EquilEqnsForSaturationPoint:
    def __init__(self, stream: ThermclcInterface, beta: float, zi,
                 max_iter=1000, tol=1e-5):
        self._stream = stream
        self.beta = beta
        self.zi = zi
        self._independent_var_values = self._create_vector_for_system_size()
        self._residual_values = self._create_vector_for_system_size()
        self._xi = self._create_vector_for_component_size()
        self._yi = self._create_vector_for_component_size()
        self._ln_phi_l_props = self._create_vector_for_component_size()
        self._ln_phi_v_props = self._create_vector_for_component_size()
        self._spec = None
        self._jac = np.zeros((self.system_size, self.system_size))
        self._max_iter = max_iter
        self._tol = tol

    @property
    def independent_vars(self):
        size = self.system_size
        k_size = size - 2
        vars = [f'lnK{i + 1}' for i in range(k_size)]
        vars.append('lnT')
        vars.append('lnP')
        return vars

    @property
    def independent_var_map(self):
        vars = self.independent_vars
        return {var: i for i, var in enumerate(vars)}

    def set_spec(self, var_name, var_value):
        var_id = self.independent_var_map[var_name]
        self._spec = (var_id, var_value)

    @property
    def system_size(self):
        size = self._stream.inflow_size + 2  # lnKi, T, P
        return size

    def _create_vector_for_system_size(self):
        return np.zeros(self.system_size)

    def _create_vector_for_component_size(self):
        inflow_size = self._stream.inflow_size
        return np.zeros(inflow_size)

    def _update_dependent_variables(self):
        self._update_xi_yi()
        self._update_phi()

    @property
    def t(self):
        return np.exp(self._independent_var_values[-2])

    @property
    def p(self):
        return np.exp(self._independent_var_values[-1])

    def _update_phi(self):
        self._ln_phi_l_props = self._stream.calc_properties(FlashInput(self.t, self.p, self._xi), PhaseEnum.LIQ)
        self._ln_phi_v_props = self._stream.calc_properties(FlashInput(self.t, self.p, self._yi), PhaseEnum.VAP)

    def _update_xi_yi(self):
        lnki = self._independent_var_values[:-2]
        ki = np.exp(lnki)
        self._xi = self.zi / (1.0 - self.beta + self.beta * ki)
        self._yi = ki * self.zi / (1.0 - self.beta + self.beta * ki)

    def _update_residuals(self):
        self._residual_values[:-2] = self._compute_ln_k_residuals()
        self._residual_values[-2] = self._compute_sum_yi_minus_xi_residual()
        self._residual_values[-1] = self._compute_spec_eqn_residual()

    def _compute_ln_k_residuals(self):
        lnki = self._independent_var_values[:-2]
        return lnki + self._ln_phi_v_props.phi - self._ln_phi_l_props.phi

    def _compute_sum_yi_minus_xi_residual(self):
        return np.sum(self._yi - self._xi)

    def _compute_spec_eqn_residual(self):
        spec_id, spec_value = self._spec
        return self._independent_var_values[spec_id] - spec_value

    def setup_independent_vars_initial_values(self, values: np.array):
        self._independent_var_values = values

    def _update_jacobian(self):
        ret = np.zeros((self.system_size, self.system_size))
        self._update_jac_dK_eqns_to_dlnK(ret)
        self._update_jac_dK_eqns_to_dlnT(ret)
        self._update_jac_dK_eqns_to_dP(ret)
        
        self._update_jac_dsum_eqn_to_dK(ret)
        self._update_jac_dspec_eqn(ret)
        self._jac = ret

    def _update_jac_dK_eqns_to_dlnK(self, ret):
        component_size = self._stream.inflow_size
        ret[:-2, :-2] += self._dlnKi_to_dlnKj(component_size)
        ret[:-2, :-2] += self._dlnphi_v_to_dlnKj()
        ret[:-2, :-2] -= self._dlnphi_l_to_dlnKj()

    def _dlnKi_to_dlnKj(self, size):
        return np.eye(size)

    def _dlnphi_v_to_dlnKj(self):
        dyi_to_dlnKj = np.diag(self._dyj_to_dlnKj)
        return np.matmul(self._ln_phi_v_props.dphi_dx, dyi_to_dlnKj)

    @property
    def _dyj_to_dlnKj(self):
        return (1.0 - self.beta) * self._xi * self._yi / self.zi

    def _dlnphi_l_to_dlnKj(self):
        dxi_to_dlnKj = np.diag(self._dxj_to_dlnKj)
        return np.matmul(self._ln_phi_l_props.dphi_dx, dxi_to_dlnKj)

    @property
    def _dxj_to_dlnKj(self):
        return - self.beta * self._xi * self._yi / self.zi

    def _update_jac_dK_eqns_to_dlnT(self, ret):
        ret[:-2, -2] += self._ln_phi_v_props.dphi_dt*self.t
        ret[:-2, -2] -= self._ln_phi_l_props.dphi_dt*self.t

    def _update_jac_dK_eqns_to_dP(self, ret):
        ret[:-2, -1] += self._ln_phi_v_props.dphi_dp*self.p
        ret[:-2, -1] -= self._ln_phi_l_props.dphi_dp*self.p

    def _update_jac_dsum_eqn_to_dK(self, ret):
        ret[-2, :-2] = self._dyj_to_dlnKj - self._dxj_to_dlnKj

    @property
    def spec_var_id(self):
        var_id, _ = self._spec
        return var_id

    def _update_jac_dspec_eqn(self, ret):
        ret[-1, self.spec_var_id] = 1.0

    @property
    def ln_k_s(self):
        return self.independent_vars[:-2]

    def solve(self, damping_factor=1.0):
        history = []
        for i in range(self._max_iter):
            self._update_dependent_variables()
            self._update_residuals()
            f_norm = np.linalg.norm(self._residual_values)
            history.append((f_norm, self._independent_var_values.copy()))
            if f_norm < self._tol:
                print(f'Newton solver for saturation point converged in {i} iterations')
                return (self.t, self.p), self.ln_k_s, i
            self._update_jacobian()
            jac_inverse = self._compute_jac_inverse()
            del_x = np.matmul(-jac_inverse, self._residual_values)
            effect_del_x = del_x*damping_factor
            effect_del_x = self._get_effective_del_x_to_avoid_overshooting(effect_del_x)
            self._independent_var_values += effect_del_x
        else:
            raise EquilEqnsForSaturationPointException(f'Newton solver did not converge in {i} iterations')

    def _compute_jac_inverse(self):
        jac_cond = np.linalg.cond(self._jac)
        if jac_cond > 1e4:
            return np.linalg.pinv(self._jac)
        else:
            return np.linalg.inv(self._jac)

    def _get_effective_del_x_to_avoid_overshooting(self, effect_del_x):
        factor = 1.0
        for i, (each_del_x, each_current_x) in enumerate(zip(effect_del_x, self._independent_var_values)):
            ff = 0.5
            if i == len(effect_del_x) - 1:    # P
                ff = 0.1
            small_current_x = ff*each_current_x
            if abs(each_del_x*factor) > abs(small_current_x):
                factor = abs(small_current_x/each_del_x)
        return factor*effect_del_x

    def compute_current_sensitivity(self):
        df_ds = np.zeros(self.system_size)
        df_ds[-1] = -1
        jac_inverse = self._compute_jac_inverse()
        sensitivity = np.matmul(jac_inverse, -df_ds)
        return sensitivity

    def solve_phase_envolope(self, t, p, manually=False):
        initial_tp, initial_ki = self._solve_from_wilson(t, p)
        ln_tp = np.log(initial_tp)
        ln_ki = np.log(initial_ki)
        self.setup_independent_vars_initial_values(np.array([*ln_ki, *ln_tp]))
        self.set_spec('lnT', np.log(t))
        _ = self.solve()
        sensitivity = self.print_var_and_sensitivity()

        while True:
            if manually:
                input_text = input()
                var_name, value = input_text.split((' '))
                value = float(value)
            self.set_spec(var_name, value)
            self._update_x_new_from_senstivity(var_name, value, sensitivity)
            self.solve()
            sensitivity = self.print_var_and_sensitivity()

    def _solve_from_wilson(self, t, p):
        wilson_solver = create_saturation_point_solver(self._stream, SaturationType.from_beta(self.beta), 'Wilson')
        tp, ki, p_iters = wilson_solver.calculate_saturation_condition(self.zi, t, p, 'P')
        return tp, ki

    def print_var_and_sensitivity(self):
        sensitivity = self.compute_current_sensitivity()
        print('Variables and sensitivity')
        for var_name, var_value, each_sensitivity in zip(self.independent_vars,
                                                         self._independent_var_values, sensitivity):
            print(f'{var_name}\t{var_value}\t{each_sensitivity}')
        print(f'T = {self.t}\tP = {self.p}')
        return sensitivity

    def _update_x_new_from_senstivity(self, var_name, value, sensitivity):
        var_id = self.independent_var_map[var_name]
        delta_s = value - self._independent_var_values[var_id]
        x_new = self._independent_var_values + sensitivity*delta_s
        self._independent_var_values = x_new

