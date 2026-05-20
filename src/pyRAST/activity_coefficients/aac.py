"""

"""
# ruff: noqa: TC002

import numpy as np
import scipy.optimize

from pyrast.activity_coefficients.activity_coefficient import ActivityCoefficient


class AAC(ActivityCoefficient, model_name='aAC'):

    # Class variables for every instance
    name = 'aAC'
    param_names = ('A_01', 'A_10', 'C')
    param_default_bounds = ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))

    def ln_gamma(self, x, phi):
        """docstring"""
        ln_gamma0 = self.model_parameters['A_01'] * \
                    (1.0 - np.exp(-self.model_parameters['C'] * phi)) * (x[1] ** 2)
        ln_gamma1 = self.model_parameters['A_10'] * \
                    (1.0 - np.exp(-self.model_parameters['C'] * phi)) * (x[0] ** 2)
        return np.array([ln_gamma0, ln_gamma1])

    def gamma(self, x, phi):
        """docstring"""
        return np.exp(self.ln_gamma(x, phi))

    def inverse_excess_loading(self, x, phi):
        """docstring"""
        A_01 = self.model_parameters['A_01']
        A_10 = self.model_parameters['A_10']
        C = self.model_parameters['C']
        return (A_01 * x[1] + A_10 * x[0]) * C * x[0] * x[1] * np.exp(-C * phi)

    def _fit_to_gamma(self):
        """docstring"""
        gamma, phi = self._gamma_from_loadings()
        x = self.comp_q / np.sum(self.comp_q)

        c = 0.2
        correction = 1.0 - np.exp(-c * phi)
        A_01 = np.log(gamma[0]) / (x[1] ** 2) / correction
        A_10 = np.log(gamma[1]) / (x[0] ** 2) / correction
        self.model_parameters = {'A_01': A_01, 'A_10': A_10, 'C': c}
        print(self.model_parameters)
