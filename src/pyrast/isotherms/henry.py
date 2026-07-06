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

    def reduced_potential(self, pressure):
        r"""Returns reduced potential as a function of pressure (or fugacity).

        Reduced potential in the Henry model is given as:

        .. math::

            \Psi(P) = K_HP

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate reduced
                potential

        Returns:
            float or np.ndarray: reduced potential as same variable type as input
        """
        return self.model_parameters['KH'] * pressure

    def p0(self, psi):
        r"""Returns P0 as a function of reduced potential.

        As the Henry model has an analytical form for P0, we can calculate it
        directly here. Activity coefficient fitting will be fastest using this model.
        P0 in the Henry model is given as:

        .. math::

            P^0(\Psi) = \frac{\Psi}{K_H}

        Args:
            psi (float or np.ndarray): Reduced potential to calculate P0

        Returns:
            float or np.ndarray: P0 as same variable type as input
        """

        return psi / self.model_parameters['KH']

    def initial_guess(self):
        """Provides initial guess for model parameters.

        For the Henry isotherm, we follow the scheme of pyIAST and assume parameter
        values based on the Langmuir model.
        """
        langmuir_guess = super().initial_guess()
        return {'KH': langmuir_guess['M'] * langmuir_guess['K']}
