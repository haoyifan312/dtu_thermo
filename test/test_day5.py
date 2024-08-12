import unittest

import numpy as np

from ThermoModelConsistencyCheck import ThermoModelConsistencyCheck
from thermclc_interface import init_system, FlashInput

example = {
    'C3': 0.2,
    'C5': 0.2,
    'C6': 0.2,
    'CO2': 0.2
}


class TestModelConsistency(unittest.TestCase):
    components = list(example.keys())
    flash_input = FlashInput(250.0, 3.0, np.array(list(example.values())))

    def test_constructor(self):
        for model in (1, 2, 3):
            with init_system(self.components, model) as _:
                self.assertTrue(True)

    def test_model1(self):
        self._test_model(1, True, True)

    def test_model2(self):
        self._test_model(2, False, True)

    def test_model3(self):
        self._test_model(3, True, False)

    def _test_model(self, mod, dg_dni_passed, dsumxphi_dp_passed):
        print(f"\n\nTesting model {mod}")
        with init_system(self.components, mod) as stream:
            thermo_check = ThermoModelConsistencyCheck(stream)
            self.assertEqual(thermo_check.test_g_der_wrt_ni(self.flash_input), dg_dni_passed)
            self.assertEqual(thermo_check.test_sum_x_phi_der_wrt_p(self.flash_input), dsumxphi_dp_passed)


