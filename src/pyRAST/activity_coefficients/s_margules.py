"""Implementation of Symmetric Margules Model"""

import numpy as np
from scipy.optimize import least_squares

from pyrast.activity_coefficients.activity_coefficient import ActivityCoefficient


class SMargules(ActivityCoefficient, model_name='sMargules'):
    r"""
    The Symmetric Margules model (also the constant temperature variant of the
    Siperstein and Myers ABC model) is the simplest activity coefficient model for
    adsorbed solutions. It is best suited for: UPDATE

    The excess Gibbs free energy in the Symmetric Margules model is given by:

    .. math:: \frac{g^E}{RT} = A x_1 x_2 (1 - e^{-C \phi})

    Sources: Siperstein, F. R. & Myers, A. L. Mixed-gas adsorption. AIChE Journal 47,
    1141-1159 (2001).

    Luberti, M., Mennitto, R., Brandani, S., Santori, G. & Sarkisov, L. Activity
    coefficient models for accurate prediction of adsorption azeotropes. Adsorption 27,
    1191-1206 (2021).
    """
    # Class variables for every instance
    name = 'sMargules'
    param_names = ('A', 'C')
    param_default_bounds = ((-np.inf, np.inf), (0.0, np.inf))

    def ln_gamma(self, x, phi):
        r"""Calculates the natural log of the activity coefficients for each component.

        In the Symmetric Margules model, the activity coefficients are calculated as:

        .. math::
            \ln \gamma_i = A x_j^2 (1 - e^{-C \phi})

        Args:
            x (array-like): Mole fractions of the components in the mixture.
            phi (float): Spreading pressure for the mixture.
        Returns:
            np.ndarray: Natural log of the activity coefficients for each component.
        """
        ln_gamma0 = self.model_parameters['A'] * \
                    (1.0 - np.exp(-self.model_parameters['C'] * phi)) * (x[1] ** 2)
        ln_gamma1 = self.model_parameters['A'] * \
                    (1.0 - np.exp(-self.model_parameters['C'] * phi)) * (x[0] ** 2)
        return np.array([ln_gamma0, ln_gamma1])

    def inverse_excess_loading(self, x, phi):
        r"""Calculates the inverse of the excess loading given composition and phi.

        The excess loading in the Symmetric Margules model is calculated as:

        .. math:: \left(\frac{1}{q}\right)^E = A C x_1 x_2 e^{-C \phi}

        Args:
            x (array-like): Mole fractions of the components in the mixture.
            phi (float): Spreading pressure for the mixture.
        Returns:
            float: Inverse of the excess loading for the mixture.
        """
        return self.model_parameters['A'] * self.model_parameters['C'] * x[0] * x[1] * \
               np.exp(-self.model_parameters['C'] * phi)

    def _fit_component_loadings(self, *, excess_loading: bool = False,
                                verbose: bool = False):
        """docstring"""
        if self.loadings.shape[0] == 1:
            # Handle the case where a single data point is provided, thus c is assumed
            gamma, phi = self._gamma_from_loadings(self.loadings, self.partial_fug,
                                                   excess_loading=excess_loading,
                                                   verbose=verbose)
            x = self.loadings / np.sum(self.loadings)
            lhs_0 = np.log(gamma[0]) / (x[1] ** 2)
            lhs_1 = np.log(gamma[1]) / (x[0] ** 2)
            lhs = (lhs_0 + lhs_1) / 2.0
            c = self.c
            correction = 1.0 - np.exp(-c * phi)
            self.model_parameters = {'A': lhs / correction, 'C': c}

        else:
            # Handle the case where multiple data points are provided
            # In this case, we can fit C and determine A as an analytical function
            points = len(self.partial_fug)
            lhs = np.zeros(points)
            phi = np.zeros(points)

            for i in range(points):
                gamma, phi[i] = self._gamma_from_loadings(self.loadings[i],
                                                          self.partial_fug[i],
                                                          excess_loading=excess_loading,
                                                          verbose=verbose)
                x = self.loadings[i] / np.sum(self.loadings[i])
                lhs_0 = np.log(gamma[0]) / (x[1] ** 2)
                lhs_1 = np.log(gamma[1]) / (x[0] ** 2)
                lhs[i] = (lhs_0 + lhs_1) / 2.0
                # maybe add warning if lhs_0 very differnet from lhs_1

            # add check to see if phi values are far apart enough

            # Fit C by minimizing least squares
            def residuals(c):
                f = phi if c <= 1e-6 else (1.0 - np.exp(-c * phi))
                a = np.dot(lhs, f) / np.dot(f, f)
                return a * f - lhs

            res = least_squares(residuals, x0=1.0, bounds=(0, np.inf),
                                xtol=self.param_tol)
            c_fit = res.x[0]
            f_fit = phi if c_fit <= 1e-6 else (1.0 - np.exp(-c_fit * phi))
            a_fit = np.dot(lhs, f_fit) / np.dot(f_fit, f_fit)

            # maybe check residuals here to be safe

            self.model_parameters = {'A': a_fit, 'C': c_fit}



