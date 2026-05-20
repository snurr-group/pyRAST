"""

"""
# ruff: noqa: TC002

import numpy as np

from pyrast.activity_coefficients.activity_coefficient import ActivityCoefficient

#from pyrast.isotherms.interpolator_isotherm import InterpolatorIsotherm
#from pyrast.isotherms.model_isotherm import ModelIsotherm


class A(ActivityCoefficient, model_name='A'):

    # Class variables for every instance
    name = 'A'
    param_names = ('A',)
    param_default_bounds = ((-np.inf, np.inf),)

    def ln_gamma(self, x, phi):
        """docstring"""
        ln_gamma0 = self.model_parameters['A'] * (x[1] ** 2)
        ln_gamma1 = self.model_parameters['A'] * (x[0] ** 2)
        return np.array([ln_gamma0, ln_gamma1])

    def gamma(self, x, phi):
        """docstring"""
        return np.exp(self.ln_gamma(x, phi))

    def _fit_to_gamma(self):
        """docstring"""
        gamma, phi = self._gamma_from_loadings()
        x = self.comp_q / np.sum(self.comp_q)
        lhs_0 = np.log(gamma[0]) / (x[1] ** 2)
        lhs_1 = np.log(gamma[1]) / (x[0] ** 2)
        print(lhs_0, lhs_1)
        #self.model_parameters = {'A': lhs_0}
        self.model_parameters = {'A': (lhs_0 + lhs_1) / 2.0}

    def inverse_excess_loading(self, x, phi):
        return 0.0
