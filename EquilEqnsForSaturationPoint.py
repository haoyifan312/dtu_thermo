import numpy as np
from matplotlib import pyplot as plt

from SaturationPointSolver import SaturationType, create_saturation_point_solver, \
    SaturationPointBySuccessiveSubstitution, SaturationPointException
from thermclc_interface import ThermclcInterface, FlashInput, PhaseEnum


class EquilEqnsForSaturationPointException(Exception):
    pass


class EquilEqnsForSaturationPoint:
    def __init__(self, stream: ThermclcInterface, beta: float, zi,
                 max_iter=1000, tol=0.01):
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
        vars.append('T')
        vars.append('P')
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
        return self._independent_var_values[-2]

    @property
    def p(self):
        return self._independent_var_values[-1]

    @property
    def lnKi(self):
        return self._independent_var_values[:-2]

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
        self._update_jac_dK_eqns_to_dT(ret)
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

    def _update_jac_dK_eqns_to_dT(self, ret):
        ret[:-2, -2] += self._ln_phi_v_props.dphi_dt
        ret[:-2, -2] -= self._ln_phi_l_props.dphi_dt

    def _update_jac_dK_eqns_to_dP(self, ret):
        ret[:-2, -1] += self._ln_phi_v_props.dphi_dp
        ret[:-2, -1] -= self._ln_phi_l_props.dphi_dp

    def _update_jac_dsum_eqn_to_dK(self, ret):
        ret[-2, :-2] = self._dyj_to_dlnKj - self._dxj_to_dlnKj

    @property
    def spec_var_id(self):
        var_id, _ = self._spec
        return var_id

    def _update_jac_dspec_eqn(self, ret):
        ret[-1, self.spec_var_id] = 1.0

    @property
    def ln_k_names(self):
        return self.independent_vars[:-2]

    def solve(self, damping_factor=1.0, accept_loose_solution=False):
        self._solve_x_new_using_successive_substitution()

        history = []
        del_x_norm = 1
        last_delx = np.zeros(self.system_size)
        best_solution = None
        for i in range(self._max_iter):
            if i == 50:
                damping_factor *= 0.5
            elif i == 100:
                damping_factor *= 0.5
            self._update_dependent_variables()
            self._update_residuals()
            f_norm = np.linalg.norm(self._residual_values)
            history.append((f_norm, del_x_norm, self._independent_var_values.copy()))
            if best_solution is None:
                best_solution = (f_norm, self._independent_var_values.copy())
            else:
                previous_best_fnorm = best_solution[0]
                if f_norm < previous_best_fnorm:
                    best_solution = (f_norm, self._independent_var_values.copy())
            if f_norm < self._tol:
                print(f'Newton solver for saturation point converged in {i} iterations')
                return (self.t, self.p), self.lnKi, i
            self._update_jacobian()
            jac_inverse = self._compute_jac_inverse()
            del_x = np.matmul(-jac_inverse, self._residual_values)
            del_x_norm = np.linalg.norm(del_x)
            effect_del_x = del_x
            effect_del_x = self._get_effective_del_x_to_avoid_overshooting(effect_del_x)
            effect_del_x = self._get_effective_del_x_to_avoid_bounding(effect_del_x)
            effect_del_x = effect_del_x*damping_factor
            if np.max(np.abs(last_delx + effect_del_x)) < 0.2: # bouncing
                effect_del_x *= 0.5
            if i > 10 and np.linalg.norm((effect_del_x)) <1e-6:
                if accept_loose_solution:
                    self._set_system_to_best_solution(best_solution)
                else:
                    raise EquilEqnsForSaturationPointException(f'newton step is stuck at iter {i} with fnorm={f_norm}')
            self._independent_var_values += effect_del_x
            last_delx = effect_del_x.copy()
        else:
            if accept_loose_solution:
                self._set_system_to_best_solution(best_solution)
            else:
                raise EquilEqnsForSaturationPointException(f'Newton solver did not converge in {i} iterations')

    def _set_system_to_best_solution(self, best_solution):
        print(f'Newton solver for phase equilibrium did not converge, set to best solution '
              f'at fnorm={best_solution[0]}')
        self._independent_var_values = best_solution[1].copy()
        self._update_dependent_variables()

    def _compute_jac_inverse(self):
        jac_cond = np.linalg.cond(self._jac)
        if jac_cond > 1e4:
            return np.linalg.pinv(self._jac)
        else:
            return np.linalg.inv(self._jac)

    def _get_effective_del_x_to_avoid_overshooting(self, effect_del_x):
        factor = 1.0
        last_two_index = list(range(self.system_size-2, self.system_size))
        for i, (each_del_x, each_current_x) in enumerate(zip(effect_del_x, self._independent_var_values)):
            ff = 0.1
            if i in last_two_index:
                small_current_x = ff*each_current_x
            else:
                small_current_x = 10.0
            if abs(each_del_x*factor) > abs(small_current_x):
                factor = abs(small_current_x/each_del_x)
        return factor*effect_del_x

    def compute_current_sensitivity(self):
        df_ds = np.zeros(self.system_size)
        df_ds[-1] = -1
        jac_inverse = self._compute_jac_inverse()
        sensitivity = np.matmul(jac_inverse, -df_ds)
        return sensitivity

    def solve_phase_envolope(self, t, p, starting_spec='T', manually=False):
        self._create_trace_for_input()
        self._setup_first_pt_initial_guess_from_successive_substitution(t, p, starting_spec)
        self._set_spec_for_first_point(p, starting_spec, t)

        _ = self.solve()
        sensitivity = self.print_var_and_sensitivity()

        t_history = []
        p_history = []
        value_history = []
        while True:
            self._plot_tp(p_history, t_history)
            value_history.append(self._independent_var_values.copy())
            if manually:
                input_text = input()
                self._trace_input(input_text)
                if input_text == 'reset':
                    self._reset_to_old_var_values(value_history[-2])
                else:
                    var_name, value = input_text.split((' '))
                    value = float(value)
                    try:
                        self.set_spec(var_name, value)
                    except KeyError as e:
                        print(f'{var_name} is not supported')
                        pass
                    self._update_x_new_from_senstivity(var_name, value, sensitivity)
            try:
                self.solve(damping_factor=0.5, accept_loose_solution=False)
            except EquilEqnsForSaturationPointException as e:
                print(e)
                self._reset_to_old_var_values(value_history[-1])
            sensitivity = self.print_var_and_sensitivity()

    def _reset_to_old_var_values(self, old_values):
        print('reset to previous point')
        self._independent_var_values = old_values

    def _set_spec_for_first_point(self, p, starting_spec, t):
        if starting_spec == 'T':
            spec_value = t
        elif starting_spec == 'P':
            spec_value = p
        self.set_spec(starting_spec, spec_value)

    def _setup_first_pt_initial_guess_from_successive_substitution(self, t, p, spec):
        tp, ki, ss_converged = self._solve_by_successive_substitution(t, p, spec, max_iter=100)
        if self._is_trivial_solution(ki) or not ss_converged:
            tp, ki = self._solve_from_wilson(t, p)

        ln_tp = np.log(tp)
        ln_ki = np.log(ki)
        self.setup_independent_vars_initial_values(np.array([*ln_ki, *tp]))

    def _solve_by_successive_substitution(self, t, p, spec, initialization_data=None, max_iter=10):
        free_var = {
            'T': 'P',
            'P': 'T'
        }
        try:
            free_var_name = free_var[spec]
        except KeyError:
            free_var_name = 'P'
        solver = self._create_successive_substitution_solver(max_iter=max_iter)
        converged = False
        try:
            tp, ki, _ = solver.solve(t, p, self.zi, free_var_name, initialization_data=initialization_data,
                                     damping_factor=0.5)
            converged = True
        except SaturationPointException as e:
            tp, ki = e.tp_ki
        return tp, ki, converged

    def _create_successive_substitution_solver(self, max_iter=10):
        if self.beta == 0.0:
            sat_type = SaturationType.BUBBLE_POINT
        elif self.beta == 1.0:
            sat_type = SaturationType.DEW_POINT
        solver = SaturationPointBySuccessiveSubstitution.create_saturation_pt_by_successive_substitution(self._stream,
                                                                                                         sat_type,
                                                                                                         max_iter=max_iter)
        return solver

    def _plot_tp(self, p_history, t_history):
        t_history.append(self.t)
        p_history.append(self.p)
        plt.plot(t_history, p_history)
        plt.ylabel('P (MPa)')
        plt.xlabel('T (K)')
        plt.savefig('phase_envelope.png')

    def _solve_from_wilson(self, t, p, free_var='P'):
        wilson_solver = create_saturation_point_solver(self._stream, SaturationType.from_beta(self.beta), 'Wilson')
        tp, ki, p_iters = wilson_solver.calculate_saturation_condition(self.zi, t, p, free_var)
        return tp, ki

    def print_var_and_sensitivity(self):
        self._update_dependent_variables()
        self._update_jacobian()
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
        factor = sensitivity[var_id]
        normalized_sensitivity = sensitivity/factor
        x_new = self._independent_var_values + normalized_sensitivity*delta_s
        self._independent_var_values = x_new

    def _create_trace_for_input(self):
        with open('trace_input.txt', 'w') as f:
            f.write('')

    def _trace_input(self, input_text):
        with open('trace_input.txt', 'a') as f:
            f.writelines(f'{input_text}\n')

    def _solve_x_new_using_successive_substitution(self):
        spec_var = self.independent_vars[self.spec_var_id]
        current_tp = [self.t, self.p]
        current_ks = np.exp(self.lnKi)
        tp, ki, ss_converged = self._solve_by_successive_substitution(self.t, self.p, spec_var,
                                                        initialization_data=(current_tp, current_ks),
                                                        max_iter=50)
        if self._is_trivial_solution(ki) or not ss_converged:
            tp, ki = self._solve_from_wilson(self.t, self.p, 'T' if spec_var == 'P' else 'P')

        if not self._is_trivial_solution(np.array(ki)) and ss_converged:
            self.setup_independent_vars_initial_values(np.array([*np.log(ki), *tp]))

    def _is_trivial_solution(self, ki):
        if np.average(np.abs(ki-1)) < 0.05:
            return True

    def _get_effective_del_x_to_avoid_bounding(self, effect_del_x):
        factor = 1.0
        while True:
            next_values = self._independent_var_values + effect_del_x*factor
            if next_values[-1] <= 0.0 or next_values[-2] <= 0.0:
                factor *= 0.5
            else:
                break
        return factor*effect_del_x

