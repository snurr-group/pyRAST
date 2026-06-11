"""Dual Site Langmuir isotherm model."""

import numpy as np

from pyrast.isotherms.model_isotherm import ModelIsotherm


class DSLangmuir(ModelIsotherm, model_name='DSLangmuir'):

    # Class variables for every instance
    name = 'DSLangmuir'
    param_names = ('M1', 'K1', 'M2', 'K2')
    param_default_bounds = ((0., np.inf), (0., np.inf), (0., np.inf), (0., np.inf))

    def loading(self, pressure):
        r"""Returns loading as a function of pressure (or fugacity).

        Loading in the Dual Site Langmuir model is given as:

        .. math::

            q(P) = M_1\frac{K_1P}{1+K_1P} + M_2\frac{K_2P}{1+K_2P}

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate loading

        Returns:
            float or np.ndarray: loading as same variable type as input
        """
        k1p = self.model_parameters["K1"] * pressure
        k2p = self.model_parameters["K2"] * pressure
        return self.model_parameters["M1"] * k1p / (1.0 + k1p) + \
               self.model_parameters["M2"] * k2p / (1.0 + k2p)

    def spreading_pressure(self, pressure):
        r"""Returns spreading pressure as a function of pressure (or fugacity).

        Spreading pressure in the Dual Site Langmuir model is given as:

        .. math::

            \phi(P) = M_1\ln(1+K_1P) + M_2\ln(1+K_2P)

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate spreading
                pressure

        Returns:
            float or np.ndarray: spreading pressure as same variable type as input
        """
        return self.model_parameters["M1"] * np.log(
            1.0 + self.model_parameters["K1"] * pressure) +\
               self.model_parameters["M2"] * np.log(
                   1.0 + self.model_parameters["K2"] * pressure)

    def initial_guess(self):
        """Provides initial guess for model parameters.

        For the Dual Site Langmuir isotherm, we follow the scheme of pyIAST and assume
        parameter values based on the Langmuir model.
        """
        langmuir_guess = super().initial_guess()
        return {
            "M1": 0.5 * langmuir_guess['M'],
            "K1": 0.4 * langmuir_guess['K'],
            "M2": 0.5 * langmuir_guess['M'],
            "K2": 0.6 * langmuir_guess['K'],
        }
