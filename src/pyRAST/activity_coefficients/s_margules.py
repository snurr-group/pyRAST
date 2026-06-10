"""

"""

import numpy as np
from scipy.optimize import least_squares

from pyrast.activity_coefficients.activity_coefficient import ActivityCoefficient


class SMargules(ActivityCoefficient, model_name='sMargules'):

    # Class variables for every instance
    name = 'sMargules'
    param_names = ('A', 'C')
    param_default_bounds = ((-np.inf, np.inf), (-np.inf, np.inf))

    def ln_gamma(self, x, phi):
        """docstring"""
        ln_gamma0 = self.model_parameters['A'] * \
                    (1.0 - np.exp(-self.model_parameters['C'] * phi)) * (x[1] ** 2)
        ln_gamma1 = self.model_parameters['A'] * \
                    (1.0 - np.exp(-self.model_parameters['C'] * phi)) * (x[0] ** 2)
        return np.array([ln_gamma0, ln_gamma1])

    def inverse_excess_loading(self, x, phi):
        """docstring"""
        return self.model_parameters['A'] * self.model_parameters['C'] * x[0] * x[1] * \
               np.exp(-self.model_parameters['C'] * phi)

    def _fit_ideal_component_loadings(self, *, excess_loading: bool = False,
                                      verbose: bool = False):
        """docstring"""
        if isinstance(self.total_f, float):
            # Handle the case where a single data point is provided, thus c is assumed
            gamma, phi = self._gamma_from_loadings(self.comp_q, self.y, self.total_f,
                                                   excess_loading=excess_loading,
                                                   verbose=verbose)
            x = self.comp_q / np.sum(self.comp_q)
            lhs_0 = np.log(gamma[0]) / (x[1] ** 2)
            lhs_1 = np.log(gamma[1]) / (x[0] ** 2)
            lhs = (lhs_0 + lhs_1) / 2.0
            c = self.c
            correction = 1.0 - np.exp(-c * phi)
            self.model_parameters = {'A': lhs / correction, 'C': c}

        else:
            # Handle the case where multiple data points are provided
            # In this case, we can fit C and determine A as an analytical function
            self.total_f = np.asarray(self.total_f) # fix pylance complaining
            points = len(self.total_f)
            lhs = np.zeros(points)
            phi = np.zeros(points)

            for i in range(points):
                gamma, phi[i] = self._gamma_from_loadings(self.comp_q[i], self.y[i],
                                                          self.total_f[i],
                                                          excess_loading=excess_loading,
                                                          verbose=verbose)
                x = self.comp_q[i] / np.sum(self.comp_q[i])
                lhs_0 = np.log(gamma[0]) / (x[1] ** 2)
                lhs_1 = np.log(gamma[1]) / (x[0] ** 2)
                lhs[i] = (lhs_0 + lhs_1) / 2.0
                # maybe add warning if lhs_0 very differnet from lhs_1

            # add check to see if phi values are far apart enough

            # Fit C by minimizing least squares
            def residuals(c):
                f = phi if c <= 1e-10 else (1.0 - np.exp(-c * phi))
                a = np.dot(lhs, f) / np.dot(f, f)
                return a * f - lhs

            res = least_squares(residuals, x0=0.2, bounds=(1e-10, np.inf),
                                ftol=self.param_tol, xtol=self.param_tol)
            c_fit = res.x[0]
            f_fit = phi if c_fit <= 1e-10 else (1.0 - np.exp(-c_fit * phi))
            a_fit = np.dot(lhs, f_fit) / np.dot(f_fit, f_fit)

            # maybe check residuals here to be safe

            self.model_parameters = {'A': a_fit, 'C': c_fit}



