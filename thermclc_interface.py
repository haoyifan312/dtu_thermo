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


def compute_wilson_k(t_k, tc_k, p_mpa, pc_mpa, omega):
    pr = pc_mpa / p_mpa
    tr = tc_k / t_k
    return pr * np.exp(5.373 * (1.0 + omega) * (1.0 - tr))


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


@dataclasses.dataclass
class ThermoResults:
    phi: np.ndarray
    dphi_dt: np.ndarray
    dphi_dp: np.ndarray
    dphi_dx: np.ndarray
    aux: np.ndarray
    phase: PhaseEnum


@dataclasses.dataclass
class FlashInput:
    T: float
    P: float
    zs: np.array


class ThermclcInterface:
    EoS = {
        'SRK': 0,
        'PR': 1
    }

    def __init__(self, inflows, eos):
        self._inflows = inflows
        self._eos = self.EoS[eos]
        self._indata()

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

    def compute_wilson_k(self, t, p, i):
        tc, pc, omega = self.get_critical_properties(i)
        return compute_wilson_k(t, tc, p, pc, omega)

    def all_wilson_ks(self, t, p):
        return [self.compute_wilson_k(t, p, i) for i in range(self.inflow_size)]


@contextlib.contextmanager
def init_system(inflows, eos):
    stream = ThermclcInterface(inflows, eos)
    yield stream
