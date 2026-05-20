"""

"""
# ruff: noqa: TC002

import numpy as np
import scipy.optimize

from pyrast.activity_coefficients.activity_coefficient import ActivityCoefficient


class AC(ActivityCoefficient, model_name='AC'):

    # Class variables for every instance
    name = 'AC'
    param_names = ('A', 'C')
    param_default_bounds = ((-np.inf, np.inf), (-np.inf, np.inf))

    def ln_gamma(self, x, phi):
        """docstring"""
        ln_gamma0 = self.model_parameters['A'] * \
                    (1.0 - np.exp(-self.model_parameters['C'] * phi)) * (x[1] ** 2)
        ln_gamma1 = self.model_parameters['A'] * \
                    (1.0 - np.exp(-self.model_parameters['C'] * phi)) * (x[0] ** 2)
        return np.array([ln_gamma0, ln_gamma1])

    def gamma(self, x, phi):
        """docstring"""
        return np.exp(self.ln_gamma(x, phi))

    def inverse_excess_loading(self, x, phi):
        """docstring"""
        return self.model_parameters['A'] * self.model_parameters['C'] * x[0] * x[1] * \
               np.exp(-self.model_parameters['C'] * phi)

    def _fit_to_gamma(self):
        """docstring"""
        gamma, phi = self._gamma_from_loadings()
        x = self.comp_q / np.sum(self.comp_q)
        lhs_0 = np.log(gamma[0]) / (x[1] ** 2)
        lhs_1 = np.log(gamma[1]) / (x[0] ** 2)
        lhs = (lhs_0 + lhs_1) / 2.0
        c = 0.3
        correction = 1.0 - np.exp(-c * phi)
        self.model_parameters = {'A': lhs / correction, 'C': c}
        print(self.model_parameters)
