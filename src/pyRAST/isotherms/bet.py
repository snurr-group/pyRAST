"""BET isotherm model."""

import numpy as np

from pyrast.isotherms.model_isotherm import ModelIsotherm


class BET(ModelIsotherm, model_name='BET'):

    # Class variables for every instance
    name = 'BET'
    param_names = ('M', 'Ka', 'Kb')
    param_default_bounds = ((0., np.inf), (0., np.inf), (0., np.inf))

    def loading(self, pressure):
        r"""Returns loading as a function of pressure (or fugacity).

        Loading in the BET model is given as:

        .. math::

            q(P) = M\frac{K_A P}{(1-K_B P)(1-K_B P+ K_A P)}

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate loading
        Returns:
            float or np.ndarray: loading as same variable type as input
        """
        return self.model_parameters['M'] * self.model_parameters['Ka'] * pressure / (
                (1.0 - self.model_parameters['Kb'] * pressure) *
                (1.0 - self.model_parameters['Kb'] * pressure +
                 self.model_parameters['Ka'] * pressure))

    def spreading_pressure(self, pressure):
        r"""Returns spreading pressure as a function of pressure (or fugacity).

        Spreading pressure in the BET model is given as:

        .. math::

            \phi(P) = M\ln(\frac{1+K_AP-K_BP}{1-K_BP})

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate spreading
                pressure
        Returns:
            float or np.ndarray: spreading pressure as same variable type as input
        """
        return self.model_parameters["M"] * np.log(
            (1.0 - self.model_parameters['Kb'] * pressure +
             self.model_parameters['Ka'] * pressure) /
            (1.0 - self.model_parameters['Kb'] * pressure))

    def initial_guess(self):
        """Provides initial guess for model parameters.

        For the BET isotherm, we follow the scheme of pyIAST and assume
        parameter values based on the Langmuir model.
        """
        langmuir_guess = super().initial_guess()
        return {'M': langmuir_guess['M'],
                'Ka': langmuir_guess['K'],
                'Kb': langmuir_guess['K'] * 0.01}
