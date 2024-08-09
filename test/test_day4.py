import unittest

import numpy as np

from StabilityAnalysis import StabilityAnalysis, SAResultType
from thermclc_interface import *


class TestStabilityAnalysis(unittest.TestCase):
    ts = [180.0, 185.0, 190.0, 203.0, 260.0, 270.0]
    ps = [4.0, 4.0, 4.0, 5.5, 6.0, 6.0]
    components = list(example_7_component.keys())
    zs = np.array(list(example_7_component.values()))

    vap_distance_gold = [0.0, 0.0204, -0.0261, -0.0020, 0.0, 0.0]
    liq_distance_gold = [0.0, 0.0, 0.0, -0.0639, 0.0618, 0.2051]

    def test_constructor(self):
        with init_system(self.components, 'SRK') as stream:
            sa = StabilityAnalysis(stream)

    def test_case_1(self):
        i = 0
        self._test_case(i)

    def test_case_2(self):
        i = 1
        self._test_case(i)

    def test_case_3(self):
        i = 2
        self._test_case(i)

    def test_case_4(self):
        i = 3
        self._test_case(i)

    def test_case_5(self):
        i = 4
        self._test_case(i)

    def test_case_6(self):
        i = 5
        self._test_case(i)

    def _test_case(self, i):
        t = self.ts[i]
        p = self.ps[i]
        flash_input = FlashInput(t, p, self.zs)
        with init_system(self.components, 'SRK') as stream:
            sa = StabilityAnalysis(stream)
            ks = stream.all_wilson_ks(t, p)
            vap_wi_guess = estimate_light_phase_from_wilson_ks(self.zs, ks)
            sa_vap_result = sa.compute(flash_input, vap_wi_guess)
            self.assertAlmostEqual(sa_vap_result.distance, self.vap_distance_gold[i], 3)
            if sa_vap_result.category == SAResultType.TRIVIAL:
                self.assertTrue(np.allclose(sa_vap_result.wi, self.zs))

            liq_wi_guess = estimate_heavy_phase_from_wilson_ks(self.zs, ks)
            sa_liq_result = sa.compute(flash_input, liq_wi_guess)
            self.assertAlmostEqual(sa_liq_result.distance, self.liq_distance_gold[i], 3)
            if sa_liq_result.category == SAResultType.TRIVIAL:
                self.assertTrue(np.allclose(sa_liq_result.wi, self.zs))
