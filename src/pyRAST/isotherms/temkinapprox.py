"""Temkin Approximation isotherm model."""

import numpy as np

from pyrast.isotherms.model_isotherm import ModelIsotherm


class TemkinApprox(ModelIsotherm, model_name='TemkinApprox'):

    # Class variables for every instance
    name = 'TemkinApprox'
    param_names = ('M', 'K', 'theta')
    param_default_bounds = ((0., np.inf), (0., np.inf), (0., np.inf))

    def loading(self, pressure):
        r"""Returns loading as a function of pressure (or fugacity).

        Loading in the Temkin Approximation model is given as:

        .. math::

            q(P) = M\frac{KP}{1+KP} + M\theta(\frac{KP}{1+KP})^2 (\frac{KP}{1+KP} -1)

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate loading

        Returns:
            float or np.ndarray: loading as same variable type as input
        """
        langmuir_frac_loading = self.model_parameters['K'] * pressure / \
                                (1.0 + self.model_parameters['K'] * pressure)
        return self.model_parameters['M'] * \
                (langmuir_frac_loading + self.model_parameters['theta'] * \
                 langmuir_frac_loading**2 * (langmuir_frac_loading - 1.0))

    def spreading_pressure(self, pressure):
        r"""Returns spreading pressure as a function of pressure (or fugacity).

        Spreading pressure in the Temkin Approximation model is given as:

        .. math::

            \phi(P) = M\ln(1+KP) + M\theta\frac{2KP+1}{2(1+KP)^2}

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate spreading
                pressure

        Returns:
            float or np.ndarray: spreading pressure as same variable type as input
        """
        one_plus_kp = 1.0 + self.model_parameters['K'] * pressure
        return self.model_parameters['M'] * \
               (np.log(one_plus_kp) + self.model_parameters['theta'] * \
                (2.0 * self.model_parameters['K'] * pressure + 1.0) \
                / (2.0 * one_plus_kp**2))

    def initial_guess(self):
        """Provides initial guess for model parameters.

        For the Temkin Approximation isotherm, we follow the scheme of pyIAST and assume
        parameter values based on the Langmuir model.
        """
        langmuir_guess = super().initial_guess()
        return {'M': langmuir_guess['M'],
                'K': langmuir_guess['K'],
                'theta': 0.0}
