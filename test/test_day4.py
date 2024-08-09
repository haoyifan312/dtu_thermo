import unittest

import numpy as np

from StabilityAnalysis import StabilityAnalysis, SAResultType
from SuccessiveSubstitutionSolver import SSAccelerationCriteriaByCycle, SSAccelerationDEM, \
    SSAccelerationCriteriaByChange
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
            acc_by_cycle = SSAccelerationCriteriaByCycle(5)
            acc_by_change = SSAccelerationCriteriaByChange(0.01)
            acc = SSAccelerationDEM(acc_by_change)

            sa = StabilityAnalysis(stream, acceleration=acc)
            ks = stream.all_wilson_ks(t, p)
            ln_phi_z = stream.calc_properties(flash_input, PhaseEnum.STABLE).phi

            vap_wi_guess = estimate_light_phase_from_wilson_ks(self.zs, ks)
            sa_vap_result, ss_iters_vap = sa.compute(flash_input, vap_wi_guess)
            print(f'Stability analysis from vapor estimate: tm={sa_vap_result.distance}, iters={ss_iters_vap}')
            self.assertAlmostEqual(sa_vap_result.distance, self.vap_distance_gold[i], 3)
            if sa_vap_result.category == SAResultType.TRIVIAL:
                self.assertTrue(np.allclose(sa_vap_result.wi, self.zs))

            vap_wi_guess_extra = estimate_light_phase_from_wilson_ks_aggressive(self.zs, ln_phi_z)
            sa_vap_result_extra, ss_iters_vap_extra = sa.compute(flash_input, vap_wi_guess_extra)
            print(f'Stability analysis from aggressive vapor estimate: tm={sa_vap_result_extra.distance}, '
                  f'iters={ss_iters_vap_extra}')
            self.assertAlmostEqual(sa_vap_result_extra.distance, self.vap_distance_gold[i], 3)
            if sa_vap_result_extra.category == SAResultType.TRIVIAL:
                self.assertTrue(np.allclose(sa_vap_result_extra.wi, self.zs))

            liq_wi_guess = estimate_heavy_phase_from_wilson_ks(self.zs, ks)
            sa_liq_result, ss_iters_liq = sa.compute(flash_input, liq_wi_guess)
            print(f'Stability analysis from vapor estimate: tm={sa_liq_result.distance}, iters={ss_iters_liq}')
            self.assertAlmostEqual(sa_liq_result.distance, self.liq_distance_gold[i], 3)
            if sa_liq_result.category == SAResultType.TRIVIAL:
                self.assertTrue(np.allclose(sa_liq_result.wi, self.zs))

            liq_wi_guess_extra = estimate_heavy_phase_from_wilson_ks_aggressive(self.zs, ks, ln_phi_z)
            sa_liq_result_extra, ss_iters_liq_extra = sa.compute(flash_input, liq_wi_guess_extra)
            print(
                f'Stability analysis from vapor estimate: tm={sa_liq_result_extra.distance}, iters={ss_iters_liq_extra}')
            self.assertAlmostEqual(sa_liq_result_extra.distance, self.liq_distance_gold[i], 3)
            if sa_liq_result_extra.category == SAResultType.TRIVIAL:
                self.assertTrue(np.allclose(sa_liq_result_extra.wi, self.zs))
