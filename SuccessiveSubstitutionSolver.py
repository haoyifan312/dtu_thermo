from enum import IntEnum

import numpy as np


class SuccessiveSubstitutionException(Exception):
    pass


class SuccSubStatus(IntEnum):
    CONTINUE = 0
    CONVERGED = 1
    OVERSHOOT = 2


class SuccessiveSubstitutionSolver:

    def __init__(self, acceleration, compute_new_g_fun,
                 converged_fun,
                 compute_for_next_iter,
                 max_iter_reached_fun, max_iter=2000, tol=1e-7):
        self._max_iter = max_iter
        self._ss_tol = tol
        self._acceleration = acceleration
        self._compute_new_g = compute_new_g_fun
        self._converged_fun = converged_fun
        self._compute_for_next_iter = compute_for_next_iter
        self._max_iter_reached_fun = max_iter_reached_fun

    def solve(self, initial_g, external_data):
        """
            solve F(g(x),x) = 0

            iter K:
            g^k = g(x^k)
            F(g^k, x) = 0 -> x^k+1

            iteratively until g not change
            """

        g_kp1 = initial_g
        g_history = []
        for i in range(self._max_iter):
            g_k = g_kp1.copy()
            g_history.append(g_k)

            g_kp1, ss_status, external_data = self._compute_new_g(external_data, self._acceleration.did)
            if ss_status == SuccSubStatus.CONVERGED:
                break
            elif ss_status == SuccSubStatus.OVERSHOOT:
                g_kp1 = g_history[-2]
            elif ss_status == SuccSubStatus.CONTINUE:
                g_kp1 = self._acceleration.check(i, g_history, g_kp1)

            g_diffs = np.abs(g_kp1 - g_k)
            if np.max(g_diffs) < self._ss_tol:
                if self._converged_fun:
                    self._converged_fun()
                break

            external_data = self._compute_for_next_iter(g_kp1, external_data)
        else:
            if self._max_iter_reached_fun:
                self._max_iter_reached_fun()
        return i, external_data


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

    def clear(self):
        pass


class SSAccelerationCriteriaByCycle:
    def __init__(self, cycle=5):
        self._cycle = cycle

    def do_extrapolation(self, current_iter, lambda_term):
        return current_iter % self._cycle == 0

    def clear(self):
        pass


class SSAccelerationCriteriaByChange:
    def __init__(self, diff_tol=0.01):
        self._history = []
        self._diff_tol = diff_tol

    def do_extrapolation(self, current_iter, lambda_term):
        self._history.append(lambda_term)
        if len(self._history) < 2:
            return False
        last_term = self._history[-2]
        if abs(last_term) < 1e-50:
            return False
        rel_diff = abs((lambda_term - last_term) / last_term)
        return rel_diff < self._diff_tol

    def clear(self):
        self._history.clear()

class SSAccelerationDEM:
    """
    successive sublimation acceleration by dominant eigenvalue method
    """

    def __init__(self, criteria=SSAccelerationCriteriaByCycle()):
        self._did = False
        self._counter = 0
        self._criteria = criteria

    def clear(self):
        self._counter = 0
        self._criteria.clear()

    def check(self, current_iter, ss_var_history, new_var):
        self._did = False
        if current_iter == 0:
            return new_var

        y_kp2 = new_var
        y_kp1 = ss_var_history[-1]
        y_k = ss_var_history[-2]
        d_k = y_kp1 - y_k
        d_kp1 = y_kp2 - y_kp1
        dk_transpose_dkp1 = np.sum(d_k * d_kp1)
        if abs(dk_transpose_dkp1) < 1e-30:
            return new_var
        lam = np.sum(d_kp1) ** 2 / dk_transpose_dkp1
        lambda_denom = 1.0 - lam
        if abs(lambda_denom) < 1e-10:
            return new_var
        lambda_term = lam / lambda_denom
        if not self._criteria.do_extrapolation(current_iter, lambda_term):
            return new_var

        y_inf = y_kp2 + d_kp1 * lambda_term

        self._did = True
        self._counter += 1
        return y_inf

    @property
    def did(self):
        return self._did

    @property
    def counter(self):
        return self._counter