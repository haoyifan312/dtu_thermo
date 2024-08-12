import unittest

import numpy as np

from SaturationPointSolver import SaturationPointSolver, SaturationType, create_saturation_point_solver
from thermclc_interface import example_7_component, init_system


class TestSaturationPointByWilsonK(unittest.TestCase):
    components = list(example_7_component.keys())
    zs = np.array(list(example_7_component.values()))

    def test_constructor(self):
        with init_system(self.components, 'SRK') as stream:
            _ = create_saturation_point_solver(stream, SaturationType.BUBBLE_POINT, 'Wilson')

    def test_wilson_bubble_point(self):
        with init_system(self.components, 'SRK') as stream:
            solver = create_saturation_point_solver(stream, SaturationType.BUBBLE_POINT, 'Wilson')
            t = 200
            p = 5.0
            bubble_p, p_iters = solver.calculate_saturation_condition(self.zs, t, p, 'P')
            print(f'Bubble point pressure at T={t} is {bubble_p} using {p_iters} iterations')
            self.assertAlmostEqual(bubble_p, 5.973609489227863)

            bubble_t, t_iters = solver.calculate_saturation_condition(self.zs, t, p, 'T')
            print(f'Bubble point temperature at P={p} is {bubble_t} using {t_iters} iterations')
            self.assertAlmostEqual(bubble_t, 193.20681234654808)

    def test_wilson_dew_point(self):
        with init_system(self.components, 'SRK') as stream:
            solver = create_saturation_point_solver(stream, SaturationType.DEW_POINT, 'Wilson')
            t = 200
            p = 5.0
            dew_p, p_iters = solver.calculate_saturation_condition(self.zs, t, p, 'P')
            print(f'Dew point pressure at T={t} is {dew_p} using {p_iters} iterations')
            self.assertAlmostEqual(dew_p, 0.041231590930519424)

            dew_t, t_iters = solver.calculate_saturation_condition(self.zs, t, p, 'T')
            print(f'Bubble point temperature at P={p} is {dew_t} using {t_iters} iterations')
            self.assertAlmostEqual(dew_t, 288.2448733082847)




