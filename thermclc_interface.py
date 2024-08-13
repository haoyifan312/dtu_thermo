import contextlib
import dataclasses
from enum import IntEnum

import numpy as np

import thermclc as th

example_7_component = {
    'C1': 0.9430,
    'C2': 0.027,
    'C3': 0.0074,
    'C4': 0.0049,
    'C5': 0.0027,
    'C6': 0.001,
    'N2': 0.014
}


def wilson_k_exp_term(t_k, tc_k, omega):
    omega_term = 5.373 * (1.0 + omega)
    exp_term = np.exp((1.0 - tc_k / t_k) * omega_term)
    return exp_term


def compute_wilson_k(t_k, tc_k, p_mpa, pc_mpa, omega):
    pr = pc_mpa / p_mpa
    return pr * wilson_k_exp_term(t_k, tc_k, omega)


def compute_wilson_k_der_t(t_k, tc_k, p_mpa, pc_mpa, omega):
    omega_term = 5.373 * (1.0 + omega)
    exp_term = wilson_k_exp_term(t_k, tc_k, omega)
    return pc_mpa * tc_k * omega_term * exp_term / p_mpa / t_k / t_k


def compute_wilson_k_der_p(t_k, tc_k, p_mpa, pc_mpa, omega):
    exp_term = wilson_k_exp_term(t_k, tc_k, omega)
    return - pc_mpa * exp_term / p_mpa / p_mpa


def estimate_light_phase_from_wilson_ks(zs, ks):
    return zs * ks


def estimate_light_phase_from_wilson_ks_aggressive(zs, ln_phi):
    return zs * np.exp(ln_phi)


def estimate_heavy_phase_from_wilson_ks(zs, ks):
    return zs / ks


def estimate_heavy_phase_from_wilson_ks_aggressive(zs, ks, ln_phi):
    return zs / ks * np.exp(ln_phi)


class PhaseEnum(IntEnum):
    LIQ = 1
    VAP = -1
    VLE = 2
    STABLE = 0

    @staticmethod
    def from_value(v: int):
        ret = {
            1: PhaseEnum.LIQ,
            -1: PhaseEnum.VAP,
        }
        return ret[v]


class PropertyType(IntEnum):
    PROPERTY = 0
    TEMPERATURE_DER = 1
    PRESSURE_DER = 2
    COMPOSITION_DER = 3


@dataclasses.dataclass
class ThermoResults:
    phi: np.ndarray
    dphi_dt: np.ndarray
    dphi_dp: np.ndarray
    dphi_dx: np.ndarray
    aux: np.ndarray
    phase: PhaseEnum

    def __sub__(self, other):
        return ThermoResults(self.phi - other.phi,
                             self.dphi_dt - other.dphi_dt,
                             self.dphi_dp - other.dphi_dp,
                             self.dphi_dx - other.dphi_dx,
                             self.aux - other.aux,
                             self.phase)

@dataclasses.dataclass
class FlashInput:
    T: float
    P: float
    zs: np.array


class ThermclcInterface:
    eos_code = {
        'SRK': 0,
        'PR': 1
    }

    def __init__(self, inflows, eos):
        self._inflows = inflows
        self._eos = self._get_eos_option(eos)
        self._indata()

    def _get_eos_option(self, eos):
        return self.eos_code[eos]

    @property
    def inflow_id(self):
        return np.array([th.NAME.index(name) + 1 for name in self._inflows]).astype(int)  # Fortran index

    @property
    def inflow_size(self):
        return len(self._inflows)

    def _indata(self):
        th.INDATA(self.inflow_size, self._eos, self.inflow_id)

    def calc_properties(self, flash_input: FlashInput, desired_phase: PhaseEnum, option=5):
        (FUG, FUGT, FUGP, FUGX, AUX, FTYPE) = th.THERMO(flash_input.T, flash_input.P, flash_input.zs, desired_phase,
                                                        option)
        return ThermoResults(FUG, FUGT, FUGP, FUGX, AUX, PhaseEnum.from_value(FTYPE))

    def get_critical_properties(self, i: int):
        return th.GETCRIT(i)

    def compute_wilson_k(self, t, p, i, property_type=PropertyType.PROPERTY):
        fun = {
            PropertyType.PROPERTY: compute_wilson_k,
            PropertyType.TEMPERATURE_DER: compute_wilson_k_der_t,
            PropertyType.PRESSURE_DER: compute_wilson_k_der_p
        }

        tc, pc, omega = self.get_critical_properties(i)
        return fun[property_type](t, tc, p, pc, omega)

    def all_wilson_ks(self, t, p, property_type=PropertyType.PROPERTY):
        return [self.compute_wilson_k(t, p, i, property_type=property_type)
                for i in range(self.inflow_size)]


class ThermclcInterfaceTestModels(ThermclcInterface):
    def _get_eos_option(self, eos: int):
        return eos

    def _indata(self):
        th.INIT(self.inflow_size, self._eos, self.inflow_id)

    def calc_properties(self, flash_input: FlashInput, _, option=5):
        return th.FUGAC(flash_input.T, flash_input.P, flash_input.zs)


@contextlib.contextmanager
def init_system(inflows, eos):
    if eos in ThermclcInterface.eos_code.keys():
        stream = ThermclcInterface(inflows, eos)
    else:
        stream = ThermclcInterfaceTestModels(inflows, eos)
    yield stream
