import unittest

import numpy as np

from thermclc_interface import *


class TestThermoFunctions(unittest.TestCase):
    def test_stream(self):
        with init_system(['C1', 'C3'], 'SRK') as stream:
            self.assertTrue(True)

    def test_thermclc(self):
        """
        test example case at 2 MPa, Tb = ~150K
        survey 100-200 K
        160 and 170 K are in two phase region
        """

        p_mpa = 2
        inflow_moles = np.array(list(example_7_component.values()))
        temperatures = np.linspace(100.0, 200.0, 11)
        two_phase_ts = (160.0, 170.0)
        with init_system(example_7_component.keys(), 'SRK') as stream:
            for t_k in temperatures:
                flash_input = FlashInput(t_k, p_mpa, inflow_moles)
                result_l = stream.calc_properties(flash_input, PhaseEnum.LIQ)
                result_v = stream.calc_properties(flash_input, PhaseEnum.VAP)
                result_stable = stream.calc_properties(flash_input, PhaseEnum.STABLE)
                if t_k  in two_phase_ts:
                    self.assertEqual(result_l.phase, PhaseEnum.LIQ)
                    self.assertEqual(result_v.phase, PhaseEnum.VAP)
                    self.assertEqual(result_stable.phase, PhaseEnum.LIQ)
                    self.assertTrue(np.allclose(result_l.phi, result_stable.phi))
                    self.assertFalse(np.allclose(result_l.phi, result_v.phi))
                elif t_k < 160.0:   # liquid
                    self.assertEqual(result_l.phase, PhaseEnum.LIQ)
                    self.assertEqual(result_v.phase, PhaseEnum.LIQ)
                    self.assertEqual(result_stable.phase, PhaseEnum.LIQ)
                    self.assertTrue(np.allclose(result_l.phi, result_v.phi))
                    self.assertTrue(np.allclose(result_l.phi, result_stable.phi))
                elif t_k > 170.0:   # vapor
                    self.assertEqual(result_l.phase, PhaseEnum.VAP)
                    self.assertEqual(result_v.phase, PhaseEnum.VAP)
                    self.assertEqual(result_stable.phase, PhaseEnum.VAP)
                    self.assertTrue(np.allclose(result_l.phi, result_v.phi))
                    self.assertTrue(np.allclose(result_l.phi, result_stable.phi))

    def test_pressure_derivative(self):
        p_mpa = 2
        t = 160.0
        inflow_moles = np.array(list(example_7_component.values()))
        with init_system(example_7_component.keys(), 'SRK') as stream:
            flash_input = FlashInput(t, p_mpa, inflow_moles)
            props = stream.calc_properties(flash_input, desired_phase=PhaseEnum.LIQ)

            pert = 1e-6
            new_p = p_mpa + pert
            new_prop = stream.calc_properties(FlashInput(t, new_p, inflow_moles), desired_phase=PhaseEnum.LIQ)

            numerical_der = (new_prop.phi - props.phi)/pert
            analytical_der = props.dphi_dp
            print(f'numerical der={numerical_der}; analytical der={analytical_der}')
            self.assertTrue(np.allclose(numerical_der, analytical_der))


