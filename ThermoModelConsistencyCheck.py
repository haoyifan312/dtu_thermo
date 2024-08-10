import numpy as np

from thermclc_interface import *


class ThermoModelConsistencyCheck:
    def __init__(self, stream: ThermclcInterface, tol=1e-6):
        self._stream = stream
        self._tol = tol

    def test_g_der_wrt_ni(self, flash_input: FlashInput):
        ln_phi, z = self._stream.calc_properties(flash_input, None)
        g = self._compute_g_from_phi(flash_input, ln_phi)
        print('ln_phi\tdGdni')
        pert = 1e-6
        diff = []
        for i, each_phi in enumerate(ln_phi):
            new_ni = flash_input.zs.copy()
            new_ni[i] += pert
            new_flash_input = FlashInput(flash_input.T, flash_input.P, new_ni)
            new_ln_phi, new_z = self._stream.calc_properties(new_flash_input, None)
            new_g = self._compute_g_from_phi(FlashInput(flash_input.T, flash_input.P, new_ni), new_ln_phi)
            dg_dni = (new_g - g)/pert
            print(f'{each_phi: .4f}\t{dg_dni: .4f}')
            diff.append(abs(dg_dni - each_phi))
        avg_diff = np.average(diff)
        print(f'average diff = {avg_diff}')
        if avg_diff > self._tol:
            print('dG/dni test failed')
            return False
        return True

    def _compute_g_from_phi(self, flash_input, ln_phi):
        ni = flash_input.zs
        g = 0.0
        for each_n, each_phi in zip(ni, ln_phi):
            g += each_n * each_phi
        return g

    def test_sum_x_phi_der_wrt_p(self, flash_input: FlashInput):
        ln_phi, z = self._stream.calc_properties(flash_input, None)
        sum_x_phi = self._sum_x_phi(flash_input, ln_phi)

        pert = 1e-5
        new_flash_input = FlashInput(flash_input.T, flash_input.P + pert, flash_input.zs)
        new_ln_phi, new_z = self._stream.calc_properties(new_flash_input, None)
        new_sum_x_phi = self._sum_x_phi(new_flash_input, new_ln_phi)

        num_der = (new_sum_x_phi - sum_x_phi)/pert
        ana_der = (z - 1)/flash_input.P
        print('\nd_xlnphi/dP\t(Z-1)/P')
        print(f'{num_der: .4f}\t{ana_der: .4f}')

        diff = abs(num_der - ana_der)
        print(f'diff = {diff}')
        if diff > self._tol:
            print('d_xlnphi/dP test failed')
            return False
        return True

    def _sum_x_phi(self, flash_input: FlashInput, ln_phi):
        x = flash_input.zs/np.sum(flash_input.zs)
        ret = 0.0
        for each_x, each_phi in zip(x, ln_phi):
            ret += each_x*each_phi
        return ret

