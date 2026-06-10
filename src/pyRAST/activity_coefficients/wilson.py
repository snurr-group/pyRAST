"""

"""

import numpy as np
from scipy.optimize import least_squares, root

from pyrast.activity_coefficients.activity_coefficient import ActivityCoefficient


class Wilson(ActivityCoefficient, model_name='Wilson'):

    # Class variables for every instance
    name = 'Wilson'
    param_names = ('L12', 'L21', 'C')
    param_default_bounds = ((0, np.inf), (0, np.inf), (0.0, np.inf))

    def ln_gamma(self, x, phi):
        """docstring"""
        l12 = self.model_parameters['L12']
        l21 = self.model_parameters['L21']
        c = self.model_parameters['C']
        f = 1.0 - np.exp(-c * phi)
        ln_gamma0 = f*(1.0 - np.log(x[0] + l12*x[1]) - (x[0]/(x[0] + x[1]*l12)) - \
                       x[1]*l21/(x[1] + x[0]*l21))
        ln_gamma1 = f*(1.0 - np.log(x[1] + l21*x[0]) - (x[1]/(x[1] + x[0]*l21)) - \
                       x[0]*l12/(x[0] + x[1]*l12))
        return np.array([ln_gamma0, ln_gamma1])

    def inverse_excess_loading(self, x, phi):
        """docstring"""
        l12 = self.model_parameters['L12']
        l21 = self.model_parameters['L21']
        c = self.model_parameters['C']

        return c * np.exp(-c * phi) * (-x[0] * np.log(x[0] + l12*x[1]) - x[1] * np.log(\
            x[1] + l21*x[0]))

    def _fit_component_loadings(self, *, excess_loading: bool = False,
                                verbose: bool = False):
        """docstring"""
        if isinstance(self.total_f, float):
            # Handle the case where a single data point is provided, thus c is assumed
            gamma, phi = self._gamma_from_loadings(self.loadings, self.y, self.total_f,
                                                   excess_loading=excess_loading,
                                                   verbose=verbose)
            x = self.loadings / np.sum(self.loadings)
            c = self.c
            ln_g = np.log(gamma)

            def equations(p):
                l12, l21 = np.exp(p)
                self.model_parameters = {'L12': l12, 'L21': l21, 'C': c}
                ln_gamma = self.ln_gamma(x, phi)
                return ln_gamma - ln_g

            sol = root(equations, x0=[0.0, 0.0], method='hybr', tol=1e-10)

            if not sol.success:
                raise ValueError(
                    f"Wilson parameter fit failed: {sol.message}. "
                    "Try a different initial guess or check data quality.",
                )

            l12, l21 = np.exp(sol.x)
            self.model_parameters = {'L12': l12, 'L21': l21, 'C': c}
        else:
            # Handle the case where multiple data points are provided
            # In this case, we fit all parameters simultaneously using least squares
            self.total_f = np.asarray(self.total_f)
            points = len(self.total_f)
            gamma = np.zeros((points, 2))
            phi = np.zeros(points)
            xs = np.zeros((points, 2))

            for i in range(points):
                gamma[i], phi[i] = self._gamma_from_loadings(self.loadings[i],
                                                             self.y[i],
                                                             self.total_f[i],
                                                          excess_loading=excess_loading,
                                                             verbose=verbose)
                xs[i] = self.loadings[i] / np.sum(self.loadings[i])

            def residuals(params):
                l12 = np.exp(params[0])
                l21 = np.exp(params[1])
                c = params[2]
                self.model_parameters = {'L12': l12, 'L21': l21, 'C': c}

                res = np.zeros(points * 2)
                for i in range(points):
                    ln_gamma_pred = self.ln_gamma(xs[i], phi[i])
                    ln_gamma_exp = np.log(gamma[i])
                    res[2*i:2*i+2] = ln_gamma_pred - ln_gamma_exp
                return res

            res = least_squares(residuals, x0=[0.0, 0.0, 0.1], xtol=self.param_tol,
                                ftol=self.param_tol)
            if not res.success:
                raise ValueError(
                    f"Wilson parameter fit failed: {res.message}. "
                    "Try a different initial guess or check data quality.",
                )

            l12, l21, c = np.exp(res.x[0]), np.exp(res.x[1]), res.x[2]
            self.model_parameters = {'L12': l12, 'L21': l21, 'C': c}

