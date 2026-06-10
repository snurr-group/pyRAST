"""

"""

import numpy as np
from scipy.optimize import least_squares

from pyrast.activity_coefficients.activity_coefficient import ActivityCoefficient


class VanLaar(ActivityCoefficient, model_name='VanLaar'):

    # Class variables for every instance
    name = 'VanLaar'
    param_names = ('A12', 'A21', 'C')
    param_default_bounds = ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))

    def ln_gamma(self, x, phi):
        """docstring"""
        a12 = self.model_parameters['A12']
        a21 = self.model_parameters['A21']
        c = self.model_parameters['C']
        f = 1.0 - np.exp(-c * phi)
        ln_gamma0 = a12 * f / (1.0 + (a12 * x[0]) / (a21 * x[1]))**2
        ln_gamma1 = a21 * f / (1.0 + (a21 * x[1]) / (a12 * x[0]))**2
        return np.array([ln_gamma0, ln_gamma1])

    def inverse_excess_loading(self, x, phi):
        """docstring"""
        a12 = self.model_parameters['A12']
        a21 = self.model_parameters['A21']
        c = self.model_parameters['C']

        return c * a12 * a21 * x[0] * x[1] * np.exp(-c * phi) / (a12*x[0] + a21*x[1])

    def _fit_ideal_component_loadings(self, *, excess_loading: bool = False,
                                      verbose: bool = False):
        """docstring"""
        if isinstance(self.total_f, float):
            # Handle the case where a single data point is provided, thus c is assumed
            gamma, phi = self._gamma_from_loadings(self.comp_q, self.y, self.total_f,
                                                   excess_loading=excess_loading,
                                                   verbose=verbose)
            x = self.comp_q / np.sum(self.comp_q)
            c = self.c
            f = 1.0 - np.exp(-c * phi)
            a12 = (np.log(gamma[0])/f) * (1.0 + \
                  (x[1] * np.log(gamma[1]))/(x[0]*np.log(gamma[0])))**2
            a21 = (np.log(gamma[1])/f) * (1.0 + \
                  (x[0] * np.log(gamma[0]))/(x[1]*np.log(gamma[1])))**2
            self.model_parameters = {'A12': a12, 'A21': a21, 'C': c}

        else:
            # Handle the case where multiple data points are provided
            # In this case, we can fit C and determine A as an analytical function
            self.total_f = np.asarray(self.total_f) # fix pylance complaining
            points = len(self.total_f)
            gamma = np.zeros((points, 2))
            phi = np.zeros(points)
            xs = np.zeros((points, 2))

            for i in range(points):
                gamma[i], phi[i] = self._gamma_from_loadings(self.comp_q[i], self.y[i],
                                                          self.total_f[i],
                                                          excess_loading=excess_loading,
                                                          verbose=verbose)
                xs[i] = self.comp_q[i] / np.sum(self.comp_q[i])

            # add check to see if phi values are far apart enough

            # Define "effective parameters" to get solutions as function of c
            def effective_parameters(i):
                ln_g = np.log(gamma[i])
                x = xs[i]
                a12_eff = ln_g[0] * (1.0 + (x[1] * ln_g[1])/(x[0]*ln_g[0]))**2
                a21_eff = ln_g[1] * (1.0 + (x[0] * ln_g[0])/(x[1]*ln_g[1]))**2
                return a12_eff, a21_eff

            a12_effs = np.asarray([effective_parameters(i)[0] for i in range(points)])
            a21_effs = np.asarray([effective_parameters(i)[1] for i in range(points)])

            # Fit C by minimizing least squares
            def residuals(c):
                f = phi if c <= 1e-6 else (1.0 - np.exp(-c * phi))
                denom = np.dot(f, f)
                a12 = np.dot(a12_effs, f) / denom
                a21 = np.dot(a21_effs, f) / denom
                res12 = a12 *f - a12_effs
                res21 = a21 *f - a21_effs
                return np.concatenate((res12, res21))

            res = least_squares(residuals, x0=0.2, bounds=(1e-6, np.inf),
                                ftol=self.param_tol, xtol=self.param_tol)
            c_fit = res.x[0]
            f_fit = phi if c_fit <= 1e-6 else (1.0 - np.exp(-c_fit * phi))
            denom = np.dot(f_fit, f_fit)
            a12_fit = np.dot(a12_effs, f_fit) / denom
            a21_fit = np.dot(a21_effs, f_fit) / denom

            # maybe check residuals here to be safe

            self.model_parameters = {'A12': a12_fit, 'A21': a21_fit, 'C': c_fit}
