"""Quadratic isotherm model."""

import numpy as np

from pyrast.isotherms.model_isotherm import ModelIsotherm


class Quadratic(ModelIsotherm, model_name='Quadratic'):

    # Class variables for every instance
    name = 'Quadratic'
    param_names = ('M', 'Ka', 'Kb')
    param_default_bounds = ((0., np.inf), (0., np.inf), (0., np.inf))

    def loading(self, pressure: float):
        r"""Returns loading as a function of pressure (or fugacity).

        Loading in the Quadratic model is given as:

        .. math::

            q(P) = M\frac{(K_a + 2 K_b P)P}{1+K_aP+K_bP^2}

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate loading
        Returns:
            float or np.ndarray: loading as same variable type as input
        """
        return self.model_parameters['M'] * \
        (self.model_parameters['Ka'] + 2.0 * self.model_parameters['Kb'] * pressure) * \
        pressure / (1.0 + self.model_parameters['Ka'] * pressure +
                     self.model_parameters['Kb'] * pressure**2)

    def spreading_pressure(self, pressure: float):
        r"""Returns spreading pressure as a function of pressure (or fugacity).

        Spreading pressure in the Quadratic model is given as:

        .. math::

            \phi(P) = M\ln(1+K_aP+K_bP^2)

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate spreading
                pressure
        Returns:
            float or np.ndarray: spreading pressure as same variable type as input
        """
        return self.model_parameters['M'] * \
                np.log(1.0 + self.model_parameters['Ka'] * pressure +
                        self.model_parameters['Kb'] * pressure**2)

    def initial_guess(self):
        """Provides initial guess for model parameters.

        For the Quadratic isotherm, we follow the scheme of pyIAST and assume
        parameter values based on the Langmuir model.
        """
        langmuir_guess = super().initial_guess()
        return {'M': langmuir_guess['M'] / 2.0,
                'Ka': langmuir_guess['K'],
                'Kb': langmuir_guess['K']**2.0}
