import numpy as np

from thermclc_interface import ThermclcInterface, FlashInput, PhaseEnum


class EquilEqnsForSaturationPoint:
    def __init__(self, stream: ThermclcInterface, beta: float, zi):
        self._stream = stream
        self.beta = beta
        self.zi = zi
        self._independent_var_values = self._create_vector_for_system_size()
        self._residual_values = self._create_vector_for_system_size()
        self._xi = self._create_vector_for_component_size()
        self._yi = self._create_vector_for_component_size()
        self._ln_phi_l = self._create_vector_for_component_size()
        self._ln_phi_v = self._create_vector_for_component_size()
        self._spec = None

    @property
    def independent_vars(self):
        size = self.system_size
        k_size = size - 2
        vars = [f'lnK{i+1}' for i in range(k_size)]
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
        size = self._stream.inflow_size + 2 # lnKi, T, P
        return size

    def _create_vector_for_system_size(self):
        return np.zeros(self.system_size)

    def _create_vector_for_component_size(self):
        inflow_size = self._stream.inflow_size
        return np.zeros(inflow_size)

    def _update_phi(self):
        t = self._independent_var_values[-2]
        p = self._independent_var_values[-1]
        self._ln_phi_l = self._stream.calc_properties(FlashInput(t, p, self._xi), PhaseEnum.LIQ).phi
        self._ln_phi_v = self._stream.calc_properties(FlashInput(t, p, self._yi), PhaseEnum.VAP).phi

    def _update_xi_yi(self):
        lnki = self._independent_var_values[:-2]
        ki = np.exp(lnki)
        self._xi = self.zi/(1.0 - self.beta + self.beta*ki)
        self._yi = ki*self.zi/(1.0 - self.beta + self.beta*ki)

    def _update_residuals(self):
        self._residual_values[:-2] = self._compute_ln_k_residuals()
        self._residual_values[-2] = self._compute_sum_yi_minus_xi_residual()
        self._residual_values[-1] = self._compute_spec_eqn_residual()

    def _compute_ln_k_residuals(self):
        lnki = self._independent_var_values[:-2]
        return lnki + self._ln_phi_v - self._ln_phi_l

    def _compute_sum_yi_minus_xi_residual(self):
        return np.sum(self._yi - self._xi)

    def _compute_spec_eqn_residual(self):
        spec_id, spec_value = self._spec
        return self._independent_var_values[spec_id] - spec_value

    def setup_independent_vars_initial_values(self, values: np.array):
        self._independent_var_values = values
