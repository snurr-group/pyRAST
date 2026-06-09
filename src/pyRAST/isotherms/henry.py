"""Henry isotherm model."""

import numpy as np

from pyrast.isotherms.model_isotherm import ModelIsotherm


class Henry(ModelIsotherm, model_name='Henry'):

    # Class variables for every instance
    name = 'Henry'
    param_names = ('KH',)
    param_default_bounds = ((0., np.inf),)

    def loading(self, pressure):
        r"""Returns loading as a function of pressure (or fugacity).

        Loading in the Henry model is given as:

        .. math::

            q(P) = K_HP

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate loading
        Returns:
            float or np.ndarray: loading as same variable type as input
        """
        return self.model_parameters['KH'] * pressure

    def spreading_pressure(self, pressure):
        r"""Returns spreading pressure as a function of pressure (or fugacity).

        Spreading pressure in the Henry model is given as:

        .. math::

            \phi(P) = K_HP

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate spreading
                pressure
        Returns:
            float or np.ndarray: spreading pressure as same variable type as input
        """
        return self.model_parameters['KH'] * pressure

    def p0(self, target_phi):
        r"""Returns P0 as a function of spreading pressure.

        As the Henry model has an analytical form for P0, we can calculate it
        directly here. Activity coefficient fitting will be fastest using this model.
        P0 in the Henry model is given as:

        .. math::

            P^0(\phi) = \frac{\phi}{K_H}

        Args:
            target_phi (float or np.ndarray): Spreading pressure to calculate P0
        Returns:
            float or np.ndarray: P0 as same variable type as input
        """

        return target_phi / self.model_parameters['KH']

    def initial_guess(self):
        """Provides initial guess for model parameters.

        For the Henry isotherm, we follow the scheme of pyIAST and assume parameter
        values based on the Langmuir model.
        """
        langmuir_guess = super().initial_guess()
        return {'KH': langmuir_guess['M'] * langmuir_guess['K']}
