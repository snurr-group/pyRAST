"""Implementation of Symmetric NRTL Model"""

import numpy as np

from pyrast.activity_coefficients.activity_coefficient import ActivityCoefficient


class SNRTL(ActivityCoefficient, model_name='sNRTL'):
    r"""
    The Symmetric NRTL model is analagous to the Symmetric NRTL model for vapor liquid
    equlibria. The Symmetric NRTL model is best suited for: UPDATE

    The excess Gibbs free energy in the Symmetric NRTL model is given by:

    .. math::
        \frac{g^E}{RT} = \frac{x_1 x_2 \tau_{12} (G_{12} - 1)}
            {x_1 G_{12} + x_2} (1 - e^{-C \phi})

        G_{12} = \exp(-\alpha \tau_{12}), \ \tau_{12} = \tau_{21}, \ \alpha = 0.3

    Sources: Kaur, H., Tun, H., Sees, M. & Chen, C.-C. Local composition activity
    coefficient model for mixed-gas adsorption equilibria. Adsorption 25, 951-964
    (2019).

    Kopatsis, A., Salinger, A. & Myers, A. L. Thermodynamics of solutions with solvent
    and solute in different pure states. AIChE Journal 34, 1275-1286 (1988).
    """
    # Class variables for every instance
    name = 'sNRTL'
    param_names = ('t12', 'C')
    param_default_bounds = ((-np.inf, np.inf), (0.0, np.inf))
    param_ideal_values = (1.0,)
    alpha = 0.3

    def ln_gamma(self, x, phi):
        r"""Calculates the natural log of the activity coefficients for each component.

        In the Symmetric NRTL model, the activity coefficients are calculated as:

        .. math::
            \ln \gamma_1 = \frac{x_2^2 \tau_{12} (G_{12} - 1)}{(x_1 G_{12} + x_2)^2}
            (1 - e^{-C \phi})

            \ln \gamma_2 = \frac{x_1^2 \tau_{21} (G_{21} - 1)}{(x_2 G_{21} + x_1)^2}
            (1 - e^{-C \phi})

        Args:
            x (array-like): Mole fractions of the components in the mixture.
            phi (float): Spreading pressure for the mixture.

        Returns:
            np.ndarray: Natural log of the activity coefficients for each component.
        """
        t12 = self.model_parameters['t12']
        t21 = t12
        c = self.model_parameters['C']
        f = 1.0 - np.exp(-c * phi)
        alpha = self.alpha
        g12 = np.exp(-alpha*t12)
        g21 = np.exp(-alpha*t21)

        ln_gamma0 = x[1]**2 * f * t12 * (g12 - 1.0) / (x[0] * g12 + x[1])**2
        ln_gamma1 = x[0]**2 * f * t21 * (g21 - 1.0) / (x[1] * g21 + x[0])**2
        return np.array([ln_gamma0, ln_gamma1])

    def inverse_excess_loading(self, x, phi):
        r"""Calculates the inverse of the excess loading given composition and phi.

        The excess loading in the Symmetric NRTL model is calculated as:

        .. math:: \left(\frac{1}{q}\right)^E = \frac{x_1 x_2 \tau_{12} (G_{12} - 1)}
            {x_1 G_{12} + x_2} C e^{-C \phi}

        Args:
            x (array-like): Mole fractions of the components in the mixture.
            phi (float): Spreading pressure for the mixture.

        Returns:
            float: Inverse of the excess loading for the mixture.
        """
        t12 = self.model_parameters['t12']
        g12 = np.exp(-self.alpha*t12)
        c = self.model_parameters['C']

        return c * x[0] * x[1] * t12 * (g12 - 1.0) * np.exp(-c * phi) / \
               (x[0]*g12 + x[1])
