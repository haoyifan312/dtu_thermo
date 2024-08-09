import dataclasses
from copy import deepcopy
from enum import IntEnum

import numpy as np

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
    def __init__(self, stream: ThermclcInterface):
        self._stream = stream
        self._max_iter = 1000
        self._ss_tol = 1e-7

    def compute(self, flash_input: FlashInput, wi_guess):
        ln_phi_z = self._stream.calc_properties(flash_input, PhaseEnum.STABLE).phi
        di = ln_phi_z + np.log(flash_input.zs)

        new_ln_ws = np.log(wi_guess)
        new_ws = wi_guess.copy()
        new_flash_input = deepcopy(flash_input)
        for i in range(self._max_iter):
            new_flash_input.zs = new_ws.copy()
            new_w_props = self._stream.calc_properties(new_flash_input, PhaseEnum.STABLE)
            bracket_term = new_ln_ws + new_w_props.phi - di - 1.0
            tm = 1.0 + np.sum(new_ws * bracket_term)

            # check tm < 0.0

            old_ln_ws = new_ln_ws.copy()
            new_ln_ws = di - new_w_props.phi
            new_ws = np.exp(new_ln_ws)

            diff = np.abs(new_ln_ws - old_ln_ws)
            if np.max(diff) < self._ss_tol:
                return SAResult(tm, new_ws)
        else:
            raise SAException(f'Successive substitution on tm reached '
                                         f'max iterations {self._max_iter}')




