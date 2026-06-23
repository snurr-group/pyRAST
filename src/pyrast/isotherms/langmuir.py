"""Langmuir isotherm model."""

import numpy as np

from pyrast.isotherms.model_isotherm import ModelIsotherm


class Langmuir(ModelIsotherm, model_name='Langmuir'):

    # Class variables for every instance
    name = 'Langmuir'
    param_names = ('M', 'K')
    param_default_bounds = ((0., np.inf), (0., np.inf))

    def loading(self, pressure):
        r"""Returns loading as a function of pressure (or fugacity).

        Loading in the Langmuir model is given as:

        .. math::

            q(P) = M\frac{KP}{1+KP}

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate loading

        Returns:
            float or np.ndarray: loading as same variable type as input
        """
        return self.model_parameters['M'] * self.model_parameters['K'] * pressure / \
                (1.0 + self.model_parameters['K'] * pressure)

    def spreading_pressure(self, pressure):
        r"""Returns spreading pressure as a function of pressure (or fugacity).

        Spreading pressure in the Langmuir model is given as:

        .. math::

            \phi(P) = M\ln(1+KP)

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate spreading
                pressure

        Returns:
            float or np.ndarray: spreading pressure as same variable type as input
        """
        return (self.model_parameters['M'] *
                np.log(1.0 + self.model_parameters['K'] * pressure))

    def p0(self, target_phi: float):
        r"""Returns P0 as a function of spreading pressure.

        As the Langmuir model has an analytical form for P0, we can calculate it
        directly here. Activity coefficient fitting will be fastest using this model.
        There are additional safeguards to ensure numerical stability at high spreading
        pressures. P0 in the Langmuir model is given as:

        .. math::

            P^0(\phi) = \frac{e^{\phi/M} - 1}{K}

        Args:
            target_phi (float): Spreading pressure to calculate P0

        Returns:
            float: P0 value
        """
        m = self.model_parameters['M']
        k = self.model_parameters['K']
        if m <= 0 or k <= 0:
            return np.nan

        x = target_phi / m

        # Small x: use expm1 for precision
        if x < 50.0:
            return np.expm1(x) / k

        # Large x: use log-space to avoid overflow
        log_p = x - np.log(k)
        log_max = np.log(np.finfo(float).max)
        if log_p >= log_max:
            return np.finfo(float).max
        return np.exp(log_p)

    def initial_guess(self):
        """Provides initial guess for model parameters."""
        return super().initial_guess()
