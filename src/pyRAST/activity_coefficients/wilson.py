"""Implementation of Wilson Model"""

import numpy as np
from scipy.optimize import least_squares, root

from pyrast.activity_coefficients.activity_coefficient import ActivityCoefficient


class Wilson(ActivityCoefficient, model_name='Wilson'):
    r"""
    The Wilson model is analagous to the Wilson model for vapor liquid equlibria.
    The Wilson model is asymmetric and is best suited for: UPDATE

    The excess Gibbs free energy in the Wilson model is given by:

    .. math::
        \frac{g^E}{RT} = [-x_1 \ln(x_1 + x_2 \Lambda_{12}) - x_2 \ln(x_2 + x_1
        \Lambda_{21})] (1 - e^{-C \phi})

    Source: Krishna, R. & van Baten, J. M. How reliable is the Real Adsorbed Solution
    Theory (RAST) for estimating ternary mixture equilibrium in microporous host
    materials? Fluid Phase Equilibria 589, 114260 (2025).
    """
    # Class variables for every instance
    name = 'Wilson'
    param_names = ('L12', 'L21', 'C')
    param_default_bounds = ((0, np.inf), (0, np.inf), (0.0, np.inf))

    def ln_gamma(self, x, phi):
        r"""Calculates the natural log of the activity coefficients for each component.

        In the Wilson model, the activity coefficients are calculated as:

        .. math::
            \ln \gamma_1 = \left(1 - \ln(x_1 + x_2 \Lambda_{12}) - \frac{x_1}{x_1 + x_2
            \Lambda_{12}} - \frac{x_2 \Lambda_{21}}{x_2 + x_1 \Lambda_{21}}\right)
            (1 - e^{-C \phi})

            \ln \gamma_2 = \left(1 - \ln(x_2 + x_1 \Lambda_{21}) - \frac{x_2}{x_2 + x_1
            \Lambda_{21}} - \frac{x_1 \Lambda_{12}}{x_1 + x_2 \Lambda_{12}}\right)
            (1 - e^{-C \phi})

        Args:
            x (array-like): Mole fractions of the components in the mixture.
            phi (float): Spreading pressure for the mixture.
        Returns:
            np.ndarray: Natural log of the activity coefficients for each component.
        """
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
        r"""Calculates the inverse of the excess loading given composition and phi.

        The excess loading in the Wilson model is calculated as:

        .. math:: \left(\frac{1}{q}\right)^E = [-x_1 \ln(x_1 + x_2 \Lambda_{12}) - x_2
            \ln(x_2 + x_1 \Lambda_{21})] C e^{-C \phi}

        Args:
            x (array-like): Mole fractions of the components in the mixture.
            phi (float): Spreading pressure for the mixture.
        Returns:
            float: Inverse of the excess loading for the mixture.
        """
        l12 = self.model_parameters['L12']
        l21 = self.model_parameters['L21']
        c = self.model_parameters['C']

        return c * np.exp(-c * phi) * (-x[0] * np.log(x[0] + l12*x[1]) - x[1] * np.log(\
            x[1] + l21*x[0]))

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
            ln_g = np.log(gamma)

            def equations(p):
                l12, l21 = np.exp(p)
                self.model_parameters = {'L12': l12, 'L21': l21, 'C': c}
                ln_gamma = self.ln_gamma(x, phi)
                return ln_gamma - ln_g

            sol = root(equations, x0=[1.0, 1.0], method='hybr', tol=1e-10)

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

            res = least_squares(residuals, x0=[1.1, 0.9, 1.0], xtol=self.param_tol)
            if not res.success:
                raise ValueError(
                    f"Wilson parameter fit failed: {res.message}. "
                    "Try a different initial guess or check data quality.",
                )

            # Print residuals if verbose
            if verbose:
                print(f'Fitted parameters: L12={np.exp(res.x[0])}, '
                      'L21={np.exp(res.x[1])}, C={res.x[2]}')
                print(f'Residual norm: {res.cost}')

            l12, l21, c = np.exp(res.x[0]), np.exp(res.x[1]), res.x[2]
            self.model_parameters = {'L12': l12, 'L21': l21, 'C': c}

