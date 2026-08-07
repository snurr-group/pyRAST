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

    def reduced_potential(self, pressure):
        r"""Returns reduced potential as a function of pressure (or fugacity).

        Reduced potential in the Langmuir model is given as:

        .. math::

            \Psi(P) = M\ln(1+KP)

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate reduced
                potential

        Returns:
            float or np.ndarray: reduced potential as same variable type as input
        """
        return (self.model_parameters['M'] *
                np.log(1.0 + self.model_parameters['K'] * pressure))

    def p0(self, psi):
        r"""Returns P0 as a function of reduced potential.

        As the Langmuir model has an analytical form for P0, we can calculate it
        directly here. Activity coefficient fitting will be fastest using this model.
        There are additional safeguards to ensure numerical stability at high reduced
        potentials. P0 in the Langmuir model is given as:

        .. math::

            P^0(\Psi) = \frac{e^{\Psi/M} - 1}{K}

        Args:
            psi (float or np.ndarray): Reduced potential to calculate P0

        Returns:
            float or np.ndarray: P0 value
        """
        m = self.model_parameters['M']
        k = self.model_parameters['K']

        psi_arr = np.asarray(psi, dtype=float)
        x = psi_arr / m

        result = np.empty_like(x)

        small_mask = x < 50.0
        large_mask = ~small_mask

        # Small x: use expm1 for precision
        result[small_mask] = np.expm1(x[small_mask]) / k

        # Large x: use log-space to avoid overflow
        if np.any(large_mask):
            log_p = x[large_mask] - np.log(k)
            log_max = np.log(np.finfo(float).max)

            overflow_mask = log_p >= log_max
            ok_mask = ~overflow_mask

            large_result = np.empty_like(log_p)
            large_result[overflow_mask] = np.finfo(float).max
            large_result[ok_mask] = np.exp(log_p[ok_mask])

            result[large_mask] = large_result

        # return scalar if input was scalar
        if np.ndim(psi) == 0:
            return result.item()
        return result

    def initial_guess(self):
        """Provides initial guess for model parameters."""
        return super().initial_guess()
