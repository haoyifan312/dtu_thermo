import unittest
from thermclc_interface import *


class TestThermoFunctions(unittest.TestCase):
    def test_stream(self):
        with init_system(['C1', 'C3'], 'SRK'):
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
                result_l = stream.calc_properties(t_k, p_mpa, inflow_moles, PhaseEnum.LIQ)
                result_v = stream.calc_properties(t_k, p_mpa, inflow_moles, PhaseEnum.VAP)
                result_stable = stream.calc_properties(t_k, p_mpa, inflow_moles, PhaseEnum.STABLE)
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
