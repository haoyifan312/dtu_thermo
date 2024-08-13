import unittest

import numpy as np

from RachfordRiceSolver import RachfordRiceBase, RachfordRiceSolverOption
from SaturationPointSolver import SaturationPointSolver, SaturationType, create_saturation_point_solver, \
    SaturationPointBySuccessiveSubstitution
from thermclc_interface import example_7_component, init_system


class TestSaturationPointByWilsonK(unittest.TestCase):
    components = list(example_7_component.keys())
    zs = np.array(list(example_7_component.values()))
    model = 'Wilson'

    def test_constructor(self):
        with init_system(self.components, 'SRK') as stream:
            _ = create_saturation_point_solver(stream, SaturationType.BUBBLE_POINT, self.model)
            _ = create_saturation_point_solver(stream, SaturationType.DEW_POINT, self.model)

    def test_wilson_bubble_point(self):
        with init_system(self.components, 'SRK') as stream:
            solver = create_saturation_point_solver(stream, SaturationType.BUBBLE_POINT, self.model)
            t = 200
            p = 5.0
            bubble_tp, _, p_iters = solver.calculate_saturation_condition(self.zs, t, p, 'P')
            bubble_p = bubble_tp[1]
            print(f'Bubble point pressure at T={t} is {bubble_p} using {p_iters} iterations')
            self.assertAlmostEqual(bubble_p, 5.973609489227863)

            bubble_tp, _, t_iters = solver.calculate_saturation_condition(self.zs, t, p, 'T')
            bubble_t = bubble_tp[0]
            print(f'Bubble point temperature at P={p} is {bubble_t} using {t_iters} iterations')
            self.assertAlmostEqual(bubble_t, 193.20681234654808)

    def test_wilson_dew_point(self):
        with init_system(self.components, 'SRK') as stream:
            solver = create_saturation_point_solver(stream, SaturationType.DEW_POINT, 'Wilson')
            t = 200
            p = 5.0
            dew_tp, _, p_iters = solver.calculate_saturation_condition(self.zs, t, p, 'P')
            dew_p = dew_tp[1]
            print(f'Dew point pressure at T={t} is {dew_p} using {p_iters} iterations')
            self.assertAlmostEqual(dew_p, 0.041231590930519424)

            dew_tp, _, t_iters = solver.calculate_saturation_condition(self.zs, t, p, 'T')
            dew_t = dew_tp[0]
            print(f'Bubble point temperature at P={p} is {dew_t} using {t_iters} iterations')
            self.assertAlmostEqual(dew_t, 288.2448733082847)


class TestSaturationPointByPhi(unittest.TestCase):
    components = list(example_7_component.keys())
    zs = np.array(list(example_7_component.values()))
    model = 'Phi'

    def test_constructor(self):
        with init_system(self.components, 'SRK') as stream:
            _ = create_saturation_point_solver(stream, SaturationType.BUBBLE_POINT, self.model)
            _ = create_saturation_point_solver(stream, SaturationType.DEW_POINT, self.model)

    def test_phi_bubble_point_p(self):
        with init_system(self.components, 'SRK') as stream:
            solver = create_saturation_point_solver(stream, SaturationType.BUBBLE_POINT, self.model)
            solver_wilson = create_saturation_point_solver(stream, SaturationType.BUBBLE_POINT, 'Wilson')
            t = 200
            p = 5.0
            bubble_tp_wilson, k_wilson, p_iters_wilson = solver_wilson.calculate_saturation_condition(self.zs, t, p,
                                                                                                      'P')

            rr_solver = RachfordRiceBase.create_solver(stream.inflow_size, RachfordRiceSolverOption.BASE)
            rr_result = rr_solver.compute(k_wilson, self.zs, 1e-6)

            solver.set_incipient_x(rr_result.ys)  # bubble point
            solver.set_zi(self.zs)

            bubble_tp, k_phi, p_iters = solver.calculate_saturation_condition(self.zs, *bubble_tp_wilson, 'P')
            bubble_p = bubble_tp[1]
            print(f'Bubble point pressure at T={t} is {bubble_p} using {p_iters} iterations')
            self.assertAlmostEqual(bubble_p, 5.2992230786926635)

    def test_phi_bubble_point_t(self):
        with init_system(self.components, 'SRK') as stream:
            solver = create_saturation_point_solver(stream, SaturationType.BUBBLE_POINT, self.model)
            solver_wilson = create_saturation_point_solver(stream, SaturationType.BUBBLE_POINT, 'Wilson')
            t = 200
            p = 5.0
            bubble_tp_wilson, k_wilson, p_iters_wilson = solver_wilson.calculate_saturation_condition(self.zs,
                                                                                                      t, p, 'T')

            rr_solver = RachfordRiceBase.create_solver(stream.inflow_size, RachfordRiceSolverOption.BASE)
            rr_result = rr_solver.compute(k_wilson, self.zs, 1e-6)

            solver.set_incipient_x(rr_result.ys)  # bubble point
            solver.set_zi(self.zs)

            bubble_tp, k_phi, p_iters = solver.calculate_saturation_condition(self.zs, *bubble_tp_wilson, 'T')
            bubble_t = bubble_tp[0]
            print(f'Bubble point pressure at T={t} is {bubble_t} using {p_iters} iterations')
            self.assertAlmostEqual(bubble_t, 196.5669134913179)

    def _test_phi_dew_point_p(self):
        with init_system(self.components, 'SRK') as stream:
            solver = create_saturation_point_solver(stream, SaturationType.DEW_POINT, self.model)
            solver_wilson = create_saturation_point_solver(stream, SaturationType.DEW_POINT, 'Wilson')
            t = 200
            p = 5.0
            bubble_tp_wilson, k_wilson, p_iters_wilson = solver_wilson.calculate_saturation_condition(self.zs, t, p,
                                                                                                      'P')

            rr_solver = RachfordRiceBase.create_solver(stream.inflow_size, RachfordRiceSolverOption.BASE)
            rr_result = rr_solver.compute(k_wilson, self.zs, 1.0 - 1e-6)

            solver.set_incipient_x(rr_result.xs)  # bubble point
            solver.set_zi(self.zs)

            dew_tp, k_phi, p_iters = solver.calculate_saturation_condition(self.zs, *bubble_tp_wilson, 'P')
            dew_p = dew_tp[1]
            print(f'Dew point pressure at T={t} is {dew_p} using {p_iters} iterations')
            # self.assertAlmostEqual(dew_p, 5.2992230786926635)


class TestSaturationPointSuccessiveSubstitution(unittest.TestCase):
    components = list(example_7_component.keys())
    zs = np.array(list(example_7_component.values()))

    def test_constructuro(self):
        with init_system(self.components, 'SRK') as stream:
            _ = SaturationPointBySuccessiveSubstitution.create_saturation_pt_by_successive_substitution(stream, SaturationType.BUBBLE_POINT)
            _ = SaturationPointBySuccessiveSubstitution.create_saturation_pt_by_successive_substitution(stream,
                                                                                                        SaturationType.DEW_POINT)

    def test_bubble_point_p(self):
        with init_system(self.components, 'SRK') as stream:
            solver = SaturationPointBySuccessiveSubstitution.create_saturation_pt_by_successive_substitution(stream,
                                                                                                             SaturationType.BUBBLE_POINT)
            t = 200
            p = 5.0
            tp, iters = solver.solve(t, p, self.zs, 'P')
            bubble_p = tp[1]
            print(f'Bubble point pressure at T={t} is {bubble_p} using {iters} iterations')
            # self.assertAlmostEqual(bubble_p, 0.3157815061017666)


    def test_bubble_point_t(self):
        with init_system(self.components, 'SRK') as stream:
            solver = SaturationPointBySuccessiveSubstitution.create_saturation_pt_by_successive_substitution(stream,
                                                                                                             SaturationType.BUBBLE_POINT)
            t = 200
            p = 5.0
            tp, iters = solver.solve(t, p, self.zs, 'T')
            bubble_t = tp[0]
            print(f'Bubble point temperature at P={p} is {bubble_t} using {iters} iterations')

    def test_dew_point_p(self):
        with init_system(self.components, 'SRK') as stream:
            solver = SaturationPointBySuccessiveSubstitution.create_saturation_pt_by_successive_substitution(stream,
                                                                                                             SaturationType.DEW_POINT)
            t = 200
            p = 5.0
            tp, iters = solver.solve(t, p, self.zs, 'P')
            bubble_p = tp[1]
            print(f'Dew point pressure at T={t} is {bubble_p} using {iters} iterations')