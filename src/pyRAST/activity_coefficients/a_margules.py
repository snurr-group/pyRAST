"""Implementation of Asymmetric Margules Model"""

import numpy as np

from pyrast.activity_coefficients.activity_coefficient import ActivityCoefficient


class AMargules(ActivityCoefficient, model_name='aMargules'):
    r"""
    The Asymmetric Margules model is an extension of the traditional Margules model that
    allows for asymmetry in the activity coefficients of the components in a
    binary mixture. It is best suited for mixtures where adsorbates differ slightly in
    size, shape, or polarity. It is more flexible than the symmetric Margules model, but
    is not great for highly non-ideal mixtures.

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
    param_ideal_values = (0.0, 0.0)

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
