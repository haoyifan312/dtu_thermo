import dataclasses
from enum import IntEnum
from typing import List
import numpy as np

from thermclc_interface import PhaseEnum


class RachfordRiceException(Exception):
    pass


@dataclasses.dataclass
class RachfordRiceResult:
    beta: float = -1.0
    xs: np.array = None
    ys: np.array = None
    betas: List = dataclasses.field(default_factory=list)

    @property
    def rr_iters(self):
        return len(self.betas)

    @property
    def phase(self):
        if 0.0 < self.beta < 1.0:
            return PhaseEnum.VLE
        elif self.beta <= 0.0:
            return PhaseEnum.LIQ
        else:
            return PhaseEnum.VAP

    def clear(self):
        if self.xs is None:
            raise RachfordRiceException('Cannot clear Rachford-Rice result that is not initialized')

        self.beta = -1.0
        self.xs[:] = 0.0
        self.ys[:] = 0.0
        self.betas.clear()

    def set_as_liquid(self, zs, beta=0.0, ys=None):
        self.beta = beta
        self.xs = zs.copy()
        if ys:
            self.ys = ys
        else:
            self.ys[:] = 0.0

    def set_as_vapor(self, zs: np.array, beta=1.0, xs=None):
        self.beta = beta
        if xs:
            self.xs = xs
        else:
            self.xs[:] = 0.0
        self.ys = zs.copy()

    def normalize_xs(self):
        self.xs = self.xs / np.sum(self.xs)

    def normalize_ys(self):
        self.ys = self.ys / np.sum(self.ys)

    @property
    def xs_or_zs(self):
        return self.xs if np.sum(self.xs) > 0.0 else self.ys

    @property
    def ys_or_zs(self):
        return self.ys if np.sum(self.ys) > 0.0 else self.xs


def raise_max_iter_exception(max_iter: int):
    raise RachfordRiceException(f'Rachford-Rice solver did not converge in {max_iter} iterations')


def max_iter_dummy(max_iter):
    pass


class RRGSolverBase:
    def __init__(self, max_iter=100, max_iter_fun=raise_max_iter_exception):
        self._ks = None
        self._zs = None
        self._max_iter = max_iter
        self._max_iter_fun = max_iter_fun

    def set_input(self, ks, zs):
        self._ks = ks
        self._zs = zs

    def solve(self, result: RachfordRiceResult, initial_guess=None):
        beta_min, beta_max = self._get_beta_min_max()

        beta_new = initial_guess
        if beta_new is None:
            beta_new = 0.5

        for i in range(self._max_iter):
            g = self.fun(beta_new)
            result.betas.append(beta_new)
            if g < 0.0:
                beta_max = beta_new
            else:
                beta_min = beta_new

            g_prime = self.der(beta_new)
            if abs(g_prime) < 1e-20:
                break
            newton_step = - g / g_prime
            if self._solver_reached_solution(g, newton_step):
                break

            beta_new += newton_step
            if beta_new > beta_max or beta_new < beta_min:
                beta_new = (beta_min + beta_max) / 2.0
        else:
            self._max_iter_fun(self._max_iter)

        self._fill_result(beta_new, result)

    def _get_beta_min_max(self):
        return 0.0, 1.0

    def need_stability_analysis(self):
        return True

    def fun(self, beta):
        ret = 0.0
        for zi, ki in zip(self._zs, self._ks):
            ret += zi * (ki - 1.0) / (1.0 - beta + beta * ki)
        return ret

    def der(self, beta):
        ret = 0.0
        for zi, ki in zip(self._zs, self._ks):
            ret -= zi * (ki - 1.0) ** 2 / (1.0 - beta + beta * ki) ** 2
        return ret

    @property
    def g0(self):
        return self.fun(0.0)

    @property
    def g1(self):
        return self.fun(1.0)

    def is_all_liquid(self):
        return self.g0 < 0.0 and self.g1 < 0.0

    def is_all_vapor(self):
        return self.g0 > 0.0 and self.g1 > 0.0

    def is_vle(self):
        return self.g0 > 0.0 > self.g1

    @staticmethod
    def _solver_reached_solution(g, newton_step):
        return abs(g) < 1e-6 and abs(newton_step) < 1e-6

    def _fill_result(self, beta, result):
        # store beta to result before capping for negative flash
        result.beta = beta

        if beta > 1.0:
            beta = 1.0
        elif beta < 0.0:
            beta = 0.0

        self._calculate_xs_ys_from_beta(beta, result)

    def _calculate_xs_ys_from_beta(self, beta, result):
        for i, (ki, zi) in enumerate(zip(self._ks, self._zs)):
            denom = 1.0 - beta + beta * ki
            result.xs[i] = zi / denom
            result.ys[i] = ki * zi / denom

    def is_there_real_beta(self):
        raise RachfordRiceException('Base g function solver does not implement real beta check')


class RRGSolverNegativeFlash(RRGSolverBase):
    @property
    def k_min(self):
        return np.min(self._ks)

    @property
    def k_max(self):
        return np.max(self._ks)

    def is_there_real_beta(self):
        return self.k_min < 1.0 < self.k_max

    def _get_beta_min_max(self):
        return -1.0 / (self.k_max - 1.0), 1.0 / (1.0 - self.k_min)

    def need_stability_analysis(self):
        return False

class RRGSolverSloppy(RRGSolverBase):
    def _fill_result(self, beta, result):
        super()._fill_result(beta, result)  # x, y are li, vi
        result.xs = result.xs * (1 - beta)
        result.ys = result.ys * beta

        sumv = sum(result.ys)
        result.xs = result.xs / (1.0 - sumv)
        result.ys = result.ys / sumv


class RachfordRiceSolverOption(IntEnum):
    BASE = 0
    SLOPPY = 1
    NEGATIVE_FLASH = 2


class RachfordRiceBase:
    def __init__(self, size, g_solver=RRGSolverBase()):
        self._size = size
        self._result = RachfordRiceResult()
        self._initialize_result()
        self._g_solver = g_solver

    def is_stability_analysis_needed(self):
        return self._g_solver.need_stability_analysis()

    @property
    def size(self):
        return self._size

    def _initialize_result(self):
        self._result.xs = np.zeros(self.size)
        self._result.ys = np.zeros(self.size)

    def compute(self, ks, zs, initial_guess=None):
        self._result.clear()
        self._check_input_sizes(ks, zs)
        self._g_solver.set_input(ks, zs)
        if self.require_solving_beta(zs):
            self._g_solver.solve(self._result, initial_guess=initial_guess)
        return self._result

    def require_solving_beta(self, zs):
        if self._g_solver.is_all_vapor():
            self._result.set_as_vapor(zs)
            return False
        elif self._g_solver.is_all_liquid():
            self._result.set_as_liquid(zs)
            return False
        else:
            return True

    def _check_input_sizes(self, ks, zs):
        if len(ks) != len(zs) or len(ks) != self.size:
            raise RachfordRiceException('Rachford-Rice input size incorrect')

    @staticmethod
    def create_solver(size: int, option: RachfordRiceSolverOption):
        if option == RachfordRiceSolverOption.BASE:
            return RachfordRiceBase(size)
        elif option == RachfordRiceSolverOption.SLOPPY:
            return RachfordRiceBase(size, RRGSolverSloppy(max_iter=1, max_iter_fun=max_iter_dummy))
        elif option == RachfordRiceSolverOption.NEGATIVE_FLASH:
            return RachfordRiceNegativeFlash(size)


class RachfordRiceNegativeFlash(RachfordRiceBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, g_solver=RRGSolverNegativeFlash())

    def require_solving_beta(self, zs):
        return self._g_solver.is_there_real_beta()
