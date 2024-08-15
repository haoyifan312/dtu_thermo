from enum import IntEnum

import numpy as np

from thermclc_interface import ThermclcInterface


class MultiPhaseSolverException(Exception):
    pass


class MultiPhaseIndexVLLE(IntEnum):
    LIQUID1 = 0
    LIQUID2 = 1
    VAPOR = 2


class MultiPhaseRachfordRice:
    def __init__(self, stream: ThermclcInterface, n_phases: int, max_iter=100, tol=1e-7):
        self._stream = stream
        self.n_phases = n_phases
        self._beta = np.zeros(n_phases)
        self.phase_active = [True] * n_phases
        self.zi = np.zeros(self.component_size)
        self._Ei = np.zeros(self.component_size)
        self._inv_phi_i_in_phase_k = np.zeros((self.component_size, n_phases))
        self._gradient_k = np.zeros(n_phases)
        self._hessian_kl = np.zeros((n_phases, n_phases))

        self._max_iter = max_iter
        self._tol = tol

    def set_beta(self, beta):
        self._beta = beta

    @property
    def active_beta_coeff(self):
        return np.array([1.0 if active else 0.0 for active in self.phase_active])

    def get_effective_beta(self):
        return self._beta * self.active_beta_coeff

    def set_zi(self, zi):
        self.zi = zi

    def set_phi_all(self, phi_i_k):
        self._inv_phi_i_in_phase_k = 1.0 / phi_i_k

    def set_phi_for_phase(self, phi: np.array, phase_k):
        self._inv_phi_i_in_phase_k[:, phase_k] = 1.0 / phi

    @property
    def component_size(self):
        return self._stream.inflow_size

    def compute_q(self):
        beta = self.get_effective_beta()
        sec_term = self.zi * np.log(self._Ei)
        return np.sum(beta) - np.sum(sec_term)

    def _update_Ei(self):
        beta = self.get_effective_beta()
        ei_as_col = np.matmul(self._inv_phi_i_in_phase_k, np.transpose(beta))
        self._Ei = np.transpose(ei_as_col)

    def _update_gradient(self):
        zi_by_Ei = self.zi / self._Ei
        sec_term = np.matmul(zi_by_Ei, self._inv_phi_i_in_phase_k)
        self._gradient_k = 1.0 - sec_term

    def _update_hessian(self):
        zi_by_Ei2 = self.zi / self._Ei / self._Ei  # 1xn
        for l in range(self.n_phases):
            for k in range(self.n_phases):
                phi_il = self._inv_phi_i_in_phase_k[:, l]
                phi_ik = self._inv_phi_i_in_phase_k[:, k]
                self._hessian_kl[k, l] = np.sum(zi_by_Ei2 * phi_ik * phi_il)
        # overwrite inactive beta to 0
        for i, active in enumerate(self.phase_active):
            if not active:
                self._hessian_kl[i, :] = np.zeros(self.n_phases)
                self._hessian_kl[:, i] = np.zeros(self.n_phases)
                self._hessian_kl[i, i] = 1.0

    def minimize_q(self):
        self._set_all_phase_active()
        for i in range(self._max_iter):
            beta = self.get_effective_beta()
            self._update_Ei()
            self._update_gradient()
            self._update_hessian()
            current_q = self.compute_q()

            hessian_inv = np.linalg.inv(self._hessian_kl)
            newton_step = np.matmul(hessian_inv, -self._gradient_k)
            newton_step *= self.active_beta_coeff
            damping_factor = self._compute_damping_to_avoid_negative_beta(newton_step, beta)
            if not self.update_beta_to_decrease_q(damping_factor, current_q, newton_step):
                self._update_gradient()  # for convergence check
                break
            self._deactive_phase_from_zero_beta()
        else:
            raise MultiPhaseSolverException(f'MRR Q minimization did not converge in {i} iterations')
        if self._minimization_converged():
            return self.compute_q(), i

    def _set_all_phase_active(self):
        for i in range(self.n_phases):
            self._set_phase_active(True, i)

    def _set_phase_active(self, active, i):
        self.phase_active[i] = active

    def _compute_damping_to_avoid_negative_beta(self, newton_step, beta):
        next_beta = beta + newton_step
        min_beta_index = np.argmin(next_beta)
        if next_beta[min_beta_index] < 0.0:
            return -beta[min_beta_index] / newton_step[min_beta_index]
        return 1.0

    def update_beta_to_decrease_q(self, damping_factor, previous_q, newton_step):
        current_beta = self.get_effective_beta()
        while True:
            effective_newton_step = damping_factor * newton_step
            if np.linalg.norm(effective_newton_step) < self._tol:
                return False
            new_beta = current_beta + effective_newton_step
            self.set_beta(new_beta)
            self._update_Ei()
            new_q = self.compute_q()
            if new_q < previous_q + 1e-10:
                return True
            damping_factor *= 0.5

    def _deactive_phase_from_zero_beta(self):
        for i, each_beta in enumerate(self._beta):
            if each_beta <= 0.0:
                self._set_phase_active(False, i)

    def _minimization_converged(self):
        for each_g, each_beta in zip(self._gradient_k, self.get_effective_beta()):
            if each_beta > 0.0:  # beta>0   g=0
                if abs(each_g) > self._tol:
                    return False
            else:
                if each_g < -self._tol:  # beta=0    g>0
                    return False
        return True
