"""Implementation of Van Laar Model"""

import numpy as np
from scipy.optimize import least_squares

from pyrast.activity_coefficients.activity_coefficient import ActivityCoefficient


class VanLaar(ActivityCoefficient, model_name='VanLaar'):
    r"""
    The Van Laar model is analagous to the Van Laar model for vapor liquid equlibria.
    The Van Laar model is asymmetric and is best suited for: UPDATE

    The excess Gibbs free energy in the Van Laar model is given by:

    .. math::
        \frac{g^E}{RT} = \frac{A_{12} A_{21} x_1 x_2}{A_{12} x_1 + A_{21} x_2}
        (1 - e^{-C \phi})

    Source: Luberti, M., Mennitto, R., Brandani, S., Santori, G. & Sarkisov, L. Activity
    coefficient models for accurate prediction of adsorption azeotropes. Adsorption 27,
    1191-1206 (2021).
    """
    # Class variables for every instance
    name = 'VanLaar'
    param_names = ('A12', 'A21', 'C')
    param_default_bounds = ((-np.inf, np.inf), (-np.inf, np.inf), (0.0, np.inf))

    def ln_gamma(self, x, phi):
        r"""Calculates the natural log of the activity coefficients for each component.

        In the Van Laar model, the activity coefficients are calculated as:

        .. math::
            \ln \gamma_1 = \frac{A_{12}}{(1 + \frac{A_{12} x_1}{A_{21} x_2})^2}
            (1 - e^{-C \phi})

            \ln \gamma_2 = \frac{A_{21}}{(1 + \frac{A_{21} x_2}{A_{12} x_1})^2}
            (1 - e^{-C \phi})

        Args:
            x (array-like): Mole fractions of the components in the mixture.
            phi (float): Spreading pressure for the mixture.
        Returns:
            np.ndarray: Natural log of the activity coefficients for each component.
        """
        a12 = self.model_parameters['A12']
        a21 = self.model_parameters['A21']
        c = self.model_parameters['C']
        f = 1.0 - np.exp(-c * phi)
        ln_gamma0 = a12 * f / (1.0 + (a12 * x[0]) / (a21 * x[1]))**2
        ln_gamma1 = a21 * f / (1.0 + (a21 * x[1]) / (a12 * x[0]))**2
        return np.array([ln_gamma0, ln_gamma1])

    def inverse_excess_loading(self, x, phi):
        r"""Calculates the inverse of the excess loading given composition and phi.

        The excess loading in the Van Laar model is calculated as:

        .. math:: \left(\frac{1}{q}\right)^E = \frac{A_{12} A_{21} x_1 x_2}{A_{12} x_1
            + A_{21} x_2} C e^{-C \phi}

        Args:
            x (array-like): Mole fractions of the components in the mixture.
            phi (float): Spreading pressure for the mixture.
        Returns:
            float: Inverse of the excess loading for the mixture.
        """
        a12 = self.model_parameters['A12']
        a21 = self.model_parameters['A21']
        c = self.model_parameters['C']

        return c * a12 * a21 * x[0] * x[1] * np.exp(-c * phi) / (a12*x[0] + a21*x[1])

    def _fit_component_loadings(self, *, excess_loading: bool = False,
                                verbose: bool = False):
        """docstring"""
        if self.loadings.shape[0] == 1:
            # Handle the case where a single data point is provided, thus c is assumed
            gamma, phi = self._gamma_from_loadings(self.loadings, self.partial_fug,
                                                   excess_loading=excess_loading,
                                                   verbose=verbose)
            x = self.loadings / np.sum(self.loadings)
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
            points = len(self.partial_fug)
            gamma = np.zeros((points, 2))
            phi = np.zeros(points)
            xs = np.zeros((points, 2))

            for i in range(points):
                gamma[i], phi[i] = self._gamma_from_loadings(self.loadings[i],
                                                             self.partial_fug[i],
                                                          excess_loading=excess_loading,
                                                             verbose=verbose)
                xs[i] = self.loadings[i] / np.sum(self.loadings[i])

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
