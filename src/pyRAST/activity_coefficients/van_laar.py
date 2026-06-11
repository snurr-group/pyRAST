"""Implementation of Van Laar Model"""

import numpy as np

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
    param_ideal_values = (1e-6, 1e-6)

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
