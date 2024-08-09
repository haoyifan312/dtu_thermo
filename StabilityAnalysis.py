import dataclasses
from copy import deepcopy
from enum import IntEnum

import numpy as np

from SuccessiveSubstitutionSolver import SuccSubStatus, SuccessiveSubstitutionSolver, SSAccelerationDummy
from thermclc_interface import FlashInput, ThermclcInterface, PhaseEnum


class SAException(Exception):
    pass


class SAResultType(IntEnum):
    TRIVIAL = 0
    POSITIVE = 1
    NEGATIVE = -1


@dataclasses.dataclass
class SAResult:
    distance: float
    wi: np.array

    @property
    def category(self):
        if abs(self.distance) < 1e-12:
            return SAResultType.TRIVIAL
        elif self.distance > 0.0:
            return SAResultType.POSITIVE
        else:
            return SAResultType.NEGATIVE


class StabilityAnalysis:
    def __init__(self, stream: ThermclcInterface, ss_max_iter=1000, ss_tol=1e-7):
        self._stream = stream
        self._ss_max_iter = ss_max_iter
        self._ss_tol = ss_tol

    def d(self, x, ln_phi_x):
        return np.log(x) + ln_phi_x

    def compute(self, flash_input: FlashInput, wi_guess):
        ln_phi_z = self._stream.calc_properties(flash_input, PhaseEnum.STABLE).phi
        di = self.d(flash_input.zs, ln_phi_z)
        new_flash_input = deepcopy(flash_input)

        def compute_phi_k(external_data, did_acceleration):
            new_ws, new_ln_ws = external_data
            new_flash_input.zs = new_ws.copy()
            new_w_props = self._stream.calc_properties(new_flash_input, PhaseEnum.STABLE)
            ln_phi_w = new_w_props.phi
            return ln_phi_w, SuccSubStatus.CONTINUE, (new_ws, new_ln_ws)

        def compute_wi(ln_phi_w, _):
            new_ln_ws = di - ln_phi_w
            new_ws = np.exp(new_ln_ws)
            return new_ws, new_ln_ws

        def max_iter_action():
            raise SAException(f'Successive substitution on tm reached '
                              f'max iterations {self._ss_max_iter}')

        ss = SuccessiveSubstitutionSolver(compute_phi_k, None, compute_wi,
                                          max_iter_action, max_iter=self._ss_max_iter,
                                          tol=self._ss_tol)
        initial_flash_input = FlashInput(flash_input.T, flash_input.T, wi_guess)
        ss_iters, (ws, ln_ws) = ss.solve(self._stream.calc_properties(initial_flash_input, PhaseEnum.STABLE).phi,
                                         (wi_guess, np.log(wi_guess)))

        flash_input_w = FlashInput(flash_input.T, flash_input.P, ws)
        ln_phi_w = self._stream.calc_properties(flash_input_w, PhaseEnum.STABLE).phi
        return SAResult(self.compute_tm(ws, ln_phi_w, di), ws), ss_iters

    def compute_tm(self, wi: np.array, ln_phi_w, di):
        dw = self.d(wi, ln_phi_w)
        bracket_term = dw - di - 1.0
        tm = 1.0 + np.sum(wi * bracket_term)
        return tm
