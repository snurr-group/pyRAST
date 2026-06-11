"""Implementation of Symmetric Margules Model"""

import numpy as np

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
    param_ideal_values = (0.0,)

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
