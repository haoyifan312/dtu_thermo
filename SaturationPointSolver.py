from enum import IntEnum
import numpy as np

from thermclc_interface import ThermclcInterface, PropertyType


class SaturationPointException(Exception):
    pass


class SaturationType(IntEnum):
    BUBBLE_POINT = 1
    DEW_POINT = 2

def bubble_point_equation(zi: np.array, ki: np.array):
    return np.sum(zi * ki) - 1.0

def bubble_point_der(zi: np.array, ki: np.array, k_ders: np.array):
    return np.sum(zi * k_ders)

def dew_point_equation(zi: np.array, ki: np.array):
    return np.sum(zi / ki) - 1.0

def dew_point_der(zi: np.array, ki: np.array, k_ders: np.array):
    return - np.sum(zi/ki/ki*k_ders)


class SaturationPointSolver:
    def __init__(self, stream: ThermclcInterface, equation_fun, der_fun, max_iter=1000, tol=1e-7):
        self._stream = stream
        self._eqn_fun = equation_fun
        self._der_fun = der_fun
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
            ki = self._compute_ki(*tp, PropertyType.PROPERTY)
            f = self._eqn_fun(zi, ki)
            if abs(f) < self._tol:
                break

            k_der = self._compute_ki(*tp, der_type)
            f_prime = self._der_fun(zi, ki, k_der)
            if abs(f_prime) < 1e-50:
                raise SaturationPointException(f'Newton derivative is {f_prime}')


            newton_step = -f/f_prime
            if abs(newton_step) < self._tol*self._tol:
                raise SaturationPointException(f'Newton step is reduced to {newton_step} and cannot converge')

            tp[free_var_index] += newton_step*damping_factor
        else:
            raise SaturationPointException(f'Newton solver cannot converge in {i} iterations')
        return tp[free_var_index], i


class SaturationPointSolverWilson(SaturationPointSolver):

    def _compute_ki(self, t, p, property_type: PropertyType):
        return self._stream.all_wilson_ks(t, p, property_type=property_type)


def create_saturation_point_solver(stream: ThermclcInterface, flash_type: SaturationType, solver_name: str):
    gov_eqn = {
        SaturationType.DEW_POINT: (dew_point_equation, dew_point_der),
        SaturationType.BUBBLE_POINT: (bubble_point_equation, bubble_point_der)
    }

    solvers = {
        'Wilson': SaturationPointSolverWilson
    }

    return solvers[solver_name](stream, *gov_eqn[flash_type])
