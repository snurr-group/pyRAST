"""Implementation of Asymmetric Margules Model"""

import numpy as np
from scipy.optimize import least_squares

from pyrast.activity_coefficients.activity_coefficient import ActivityCoefficient


class AMargules(ActivityCoefficient, model_name='aMargules'):
    r"""
    The Asymmetric Margules model is an extension of the traditional Margules model that
    allows for asymmetry in the activity coefficients of the components in a
    binary mixture. It is best suited for: UPDATE

    The excess Gibbs free energy in the Asymmetric Margules model is given by:

    .. math:: \frac{g^E}{RT} = x_1 x_2 (A_{12} x_2 + A_{21} x_1) (1 - e^{-C \phi})

    Source: Krishna, R. & van Baten, J. M. How reliable is the Real Adsorbed Solution
    Theory (RAST) for estimating ternary mixture equilibrium in microporous host
    materials? Fluid Phase Equilibria 589, 114260 (2025).
    """
    # Class variables for every instance
    name = 'aMargules'
    param_names = ('A12', 'A21', 'C')
    param_default_bounds = ((-np.inf, np.inf), (-np.inf, np.inf), (0.0, np.inf))

    def ln_gamma(self, x, phi):
        r"""Calculates the natural log of the activity coefficients for each component.

        In the Asymmetric Margules model, the activity coefficients are calculated as:

        .. math::
            \ln \gamma_1 = x_2^2 (A_{12} + 2(A_{21} - A_{12}) x_1) (1 - e^{-C \phi})

            \ln \gamma_2 = x_1^2 (A_{21} + 2(A_{12} - A_{21}) x_2) (1 - e^{-C \phi})

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
        ln_gamma0 = x[1]**2 * f * (a12 + 2*(a21 - a12)*x[0])
        ln_gamma1 = x[0]**2 * f * (a21 + 2*(a12 - a21)*x[1])
        return np.array([ln_gamma0, ln_gamma1])

    def inverse_excess_loading(self, x, phi):
        r"""Calculates the inverse of the excess loading given composition and phi.

        The excess loading in the Asymmetric Margules model is calculated as:

        .. math::
            \left(\frac{1}{q}\right)^E = C x_1 x_2 (A_{12} x_2 + A_{21} x_1)
            e^{-C \phi}

        Args:
            x (array-like): Mole fractions of the components in the mixture.
            phi (float): Spreading pressure for the mixture.
        Returns:
            float: Inverse of the excess loading for the mixture.
        """
        a12 = self.model_parameters['A12']
        a21 = self.model_parameters['A21']
        c = self.model_parameters['C']

        return c * x[0] * x[1] * np.exp(-c * phi) * (a12*x[1] + a21*x[0])

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
            a12 = (2 * x[1]**2 * np.log(gamma[1]) + 2 * x[0] * x[1] * np.log(gamma[0]) \
                   - x[0] * np.log(gamma[0])) / (f * x[0] * x[1]**2 * (2 * x[0] + 2 * \
                                                                       x[1] - 1))
            a21 = (2 * x[0]**2 * np.log(gamma[0]) + 2 * x[0] * x[1] * np.log(gamma[1]) \
                   - x[1] * np.log(gamma[1])) / (f * x[1] * x[0]**2 * (2 * x[0] + 2 * \
                                                                       x[1] - 1))
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

            # Check that phi values are sufficiently different to fit parameters
            if np.max(phi) - np.min(phi) < 1e-4:
                raise ValueError('Phi values are too close together to reliably fit'
                                 'parameters. Try providing data with a wider range '
                                 'of spreading pressures.')

            # Define "effective parameters" to get solutions as function of c
            def effective_parameters(i):
                ln_g = np.log(gamma[i])
                x = xs[i]
                a12_eff = (2 * x[1]**2 * ln_g[1] + 2 * x[0] * x[1] * \
                           ln_g[0] - x[0] * ln_g[0]) / \
                            (x[0] * x[1]**2 * (2 * x[0] + 2 * x[1] - 1))
                a21_eff = (2 * x[0]**2 * ln_g[0] + 2 * x[1] * x[0] * \
                           ln_g[1] - x[1] * ln_g[1]) / \
                            (x[1] * x[0]**2 * (2 * x[0] + 2 * x[1] - 1))
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

            res = least_squares(residuals, x0=1, bounds=(0, np.inf),
                                xtol=self.param_tol)
            c_fit = res.x[0]
            f_fit = phi if c_fit <= 1e-6 else (1.0 - np.exp(-c_fit * phi))
            denom = np.dot(f_fit, f_fit)
            a12_fit = np.dot(a12_effs, f_fit) / denom
            a21_fit = np.dot(a21_effs, f_fit) / denom

            # Print residuals if verbose
            if verbose:
                print(f'Fitted parameters: A12={a12_fit}, A21={a21_fit}, C={c_fit}')
                print(f'Residual norm: {res.cost}')

            self.model_parameters = {'A12': a12_fit, 'A21': a21_fit, 'C': c_fit}
