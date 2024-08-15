import unittest

import numpy as np

from EquilEqnsForSaturationPoint import EquilEqnsForSaturationPoint
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
            _ = SaturationPointBySuccessiveSubstitution.create_saturation_pt_by_successive_substitution(stream,
                                                                                                        SaturationType.BUBBLE_POINT)
            _ = SaturationPointBySuccessiveSubstitution.create_saturation_pt_by_successive_substitution(stream,
                                                                                                        SaturationType.DEW_POINT)

    def test_bubble_point_p(self):
        with init_system(self.components, 'SRK') as stream:
            solver = SaturationPointBySuccessiveSubstitution.create_saturation_pt_by_successive_substitution(stream,
                                                                                                             SaturationType.BUBBLE_POINT)
            t = 200
            p = 5.0
            tp, _, iters = solver.solve(t, p, self.zs, 'P', damping_factor=0.5)
            bubble_p = tp[1]
            print(f'Bubble point pressure at T={t} is {bubble_p} using {iters} iterations')
            self.assertAlmostEqual(bubble_p, 5.504670651878922)

    def test_bubble_point_t(self):
        with init_system(self.components, 'SRK') as stream:
            solver = SaturationPointBySuccessiveSubstitution.create_saturation_pt_by_successive_substitution(stream,
                                                                                                             SaturationType.BUBBLE_POINT)
            t = 200
            p = 5.0
            tp, _, iters = solver.solve(t, p, self.zs, 'T', plot_t_vs_k6='BubbleT5P.png')
            bubble_t = tp[0]
            print(f'Bubble point temperature at P={p} is {bubble_t} using {iters} iterations')
            self.assertAlmostEqual(bubble_t, 195.8022784575512, 4)

    def test_bubble_point_t_random(self):
        with init_system(self.components, 'SRK') as stream:
            solver = SaturationPointBySuccessiveSubstitution.create_saturation_pt_by_successive_substitution(stream,
                                                                                                             SaturationType.BUBBLE_POINT)
            t = 200
            p = 5.9
            tp, _, iters = solver.solve(t, p, self.zs, 'T')
            bubble_t = tp[0]
            print(f'Bubble point temperature at P={p} is {bubble_t} using {iters} iterations')

    def test_dew_point_p(self):
        with init_system(self.components, 'SRK') as stream:
            solver = SaturationPointBySuccessiveSubstitution.create_saturation_pt_by_successive_substitution(stream,
                                                                                                             SaturationType.DEW_POINT)
            t = 200
            p = 5.0
            tp, _, iters = solver.solve(t, p, self.zs, 'P', damping_factor=0.5)  # had to introduce damping
            bubble_p = tp[1]
            print(f'Dew point pressure at T={t} is {bubble_p} using {iters} iterations')
            self.assertAlmostEqual(bubble_p, 0.011222187243660869)

    def test_dew_point_t(self):
        with init_system(self.components, 'SRK') as stream:
            solver = SaturationPointBySuccessiveSubstitution.create_saturation_pt_by_successive_substitution(stream,
                                                                                                             SaturationType.DEW_POINT)
            t = 200
            p = 5.0
            tp, _, iters = solver.solve(t, p, self.zs, 'T', plot_t_vs_k6='DewT5P.png')  # had to introduce damping
            bubble_t = tp[0]
            print(f'Dew point pressure at P={p} is {bubble_t} using {iters} iterations')
            self.assertAlmostEqual(bubble_t, 259.18159068192284, 4)

    def test_dew_point_t_random(self):
        with init_system(self.components, 'SRK') as stream:
            solver = SaturationPointBySuccessiveSubstitution.create_saturation_pt_by_successive_substitution(stream,
                                                                                                             SaturationType.DEW_POINT)
            t = 200
            p = 7.4
            # tp, iters = solver.solve(t, p, self.zs, 'T', plot_t_vs_k6=f'DewT{p: .2f}P.png')    # had to introduce damping
            tp, _, iters = solver.solve(t, p, self.zs, 'T')
            bubble_t = tp[0]
            print(f'Dew point temperature at P={p} is {bubble_t} using {iters} iterations')


class TestEquilEqns(unittest.TestCase):
    components = list(example_7_component.keys())
    zs = np.array(list(example_7_component.values()))

    def test_residuals(self):
        with init_system(self.components, 'SRK') as stream:
            solver = SaturationPointBySuccessiveSubstitution.create_saturation_pt_by_successive_substitution(stream,
                                                                                                             SaturationType.BUBBLE_POINT)
            t = 200
            p = 5.0
            tp, ki, iters = solver.solve(t, p, self.zs, 'T')

            equil_eqns = EquilEqnsForSaturationPoint(stream, 0.0, self.zs)
            vars = np.zeros(stream.inflow_size + 2)
            ln_ki = np.log(ki)
            vars[:-2] = ln_ki
            vars[-2:] = tp
            equil_eqns.setup_independent_vars_initial_values(vars)
            equil_eqns.set_spec('P', p)
            equil_eqns._update_xi_yi()
            equil_eqns._update_phi()
            equil_eqns._update_residuals()
            print(equil_eqns._residual_values)
            self.assertTrue(np.allclose(equil_eqns._residual_values, np.zeros(equil_eqns.system_size), atol=2e-3))

    def test_solve(self):
        with init_system(self.components, 'SRK') as stream:
            solver = create_saturation_point_solver(stream, SaturationType.BUBBLE_POINT, 'Wilson')
            t = 100.0
            p = 0.01
            bubble_tp, ki, p_iters = solver.calculate_saturation_condition(self.zs, t, p, 'P')
            ln_ki = np.log(ki)
            initial_vars = np.array([*ln_ki, *bubble_tp])

            equil_eqns = EquilEqnsForSaturationPoint(stream, 0.0, self.zs)
            equil_eqns.setup_independent_vars_initial_values(initial_vars)
            equil_eqns.set_spec('T', t)
            tp_newton, final_ki, iters = equil_eqns.solve()
            final_t, final_p = tp_newton
            print(f'Bubble point pressure for T={t} is {final_p}, used {iters} iterations')
            self.assertAlmostEqual(final_p, 0.05089767403032754, 3)

    def test_solve2(self):
        with init_system(self.components, 'SRK') as stream:
            solver = create_saturation_point_solver(stream, SaturationType.BUBBLE_POINT, 'Wilson')

            t = 150
            p = 0.5
            bubble_tp, ki, p_iters = solver.calculate_saturation_condition(self.zs, t, p, 'P')
            ln_ki = np.log(ki)
            lntp = np.log(bubble_tp)
            initial_vars = np.array([*ln_ki, *bubble_tp])

            equil_eqns = EquilEqnsForSaturationPoint(stream, 0.0, self.zs)
            equil_eqns.setup_independent_vars_initial_values(initial_vars)
            equil_eqns.set_spec('T', t)
            tp_newton, final_ki, iters = equil_eqns.solve()
            final_t, final_p = tp_newton
            print(f'Bubble point pressure for T={t} is {final_p}, used {iters} iterations')
            self.assertAlmostEqual(final_p, 1.109529499491139, 2)

            var_names = equil_eqns.independent_vars.copy()
            sensitivity = equil_eqns.compute_current_sensitivity()
            sen = [(var, value) for var, value in zip(var_names, sensitivity)]
            pass

    def _test_solve_phase_envolope_manually(self):
        with init_system(self.components, 'SRK') as stream:
            equil_eqns = EquilEqnsForSaturationPoint(stream, 0.0, self.zs)
            t = 150
            p = 0.5
            equil_eqns.solve_phase_envolope(t, p, starting_spec='T', manually=True)

    def test_solve_dew(self):
        with init_system(self.components, 'SRK') as stream:
            solver = create_saturation_point_solver(stream, SaturationType.DEW_POINT, 'Wilson')

            t = 250
            p = 9
            dew_tp, ki, p_iters = solver.calculate_saturation_condition(self.zs, t, p, 'P')
            ln_ki = np.log(ki)
            lntp = np.log(dew_tp)
            initial_vars = np.array([*ln_ki, *dew_tp])

            equil_eqns = EquilEqnsForSaturationPoint(stream, 1.0, self.zs)
            equil_eqns.setup_independent_vars_initial_values(initial_vars)
            equil_eqns.set_spec('T', t)
            tp_newton, final_ki, iters = equil_eqns.solve()
            final_t, final_p = tp_newton
            print(f'Dew point pressure for T={t} is {final_p}, used {iters} iterations')
            self.assertAlmostEqual(final_p, 1.125205365275533, 3)

    def test_jacobian(self):
        with init_system(self.components, 'SRK') as stream:
            solver = create_saturation_point_solver(stream, SaturationType.BUBBLE_POINT, 'Wilson')
            t = 150
            p = 0.01
            bubble_tp, ki, p_iters = solver.calculate_saturation_condition(self.zs, t, p, 'P')
            ln_ki = np.log(ki)
            initial_vars = np.array([*ln_ki, t, p])

            equil_eqns = EquilEqnsForSaturationPoint(stream, 0.0, self.zs)
            equil_eqns.setup_independent_vars_initial_values(initial_vars)
            equil_eqns.set_spec('T', t)

            equil_eqns._update_dependent_variables()
            equil_eqns._update_residuals()
            equil_eqns._update_jacobian()

            jac_analytical = equil_eqns._jac.copy()

            jac_numerical = np.zeros((equil_eqns.system_size, equil_eqns.system_size))
            residuals_old = equil_eqns._residual_values.copy()
            var_old = equil_eqns._independent_var_values.copy()
            pert = 1e-6
            for i in range(equil_eqns.system_size):
                pert_new = pert
                if i in (7, 8):
                    pert_new = pert / 10
                equil_eqns._independent_var_values = var_old.copy()
                equil_eqns._independent_var_values[i] += pert_new
                equil_eqns._update_dependent_variables()
                equil_eqns._update_residuals()
                jac_numerical[:, i] = (equil_eqns._residual_values - residuals_old) / pert_new
                if not np.allclose(jac_analytical[:, i], jac_numerical[:, i], atol=1e-5):
                    print(i)
                    print(np.abs(jac_analytical[:, i] - jac_numerical[:, i]))
                self.assertTrue(np.allclose(jac_analytical[:, i], jac_numerical[:, i], atol=1e-5))
        pass
