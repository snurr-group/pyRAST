"""Implementation of Wilson Model"""

import numpy as np

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
    param_ideal_values = (1.0, 1.0)

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
