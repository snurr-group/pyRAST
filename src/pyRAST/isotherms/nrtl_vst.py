"""Adsorption NRTL Vacancy Solution Theory isotherm model."""

import numpy as np
from scipy.optimize import root

from pyrast.isotherms.model_isotherm import ModelIsotherm


class NRTL_VST(ModelIsotherm, model_name='NRTL-VST'):

    # Class variables for every instance
    name = 'NRTL-VST'
    param_names = ('M', 'K', 't10')
    param_default_bounds = ((0., np.inf), (0., np.inf), (-np.inf, np.inf))

    def pressure(self, loading):
        r"""UPDATE"""
        m = self.model_parameters['M']
        k = self.model_parameters['K']
        t10 = self.model_parameters['t10']

        t01 = -t10
        cov = loading / m
        langmuir = m * cov / (k * (1.0 - cov))
        g10 = np.exp(-0.3 * t10)
        ln_gamma1 = (1.0 - cov)**2 * t10 * (g10 - 1.0) / (1.0 - cov + cov*g10)**2
        g01 = np.exp(-0.3 * t01)
        ln_gamma0 = cov**2 * t01 * (g01 - 1.0) / (cov + (1.0 - cov)*g01)**2
        return langmuir * np.exp(ln_gamma1) / np.exp(ln_gamma0)

    def loading(self, pressure):
        r"""Returns loading as a function of pressure (or fugacity).

        UPDATE DESCRIPTION

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate loading

        Returns:
            float or np.ndarray: loading as same variable type as input
        """
        pressure_array = np.asarray(pressure)
        scalar_input = pressure_array.shape == ()
        pressure_values = np.atleast_1d(pressure_array).astype(float)

        def solve_single(target_pressure):
            if target_pressure <= 0:
                return 0.0

            def fun(x):
                return self.pressure(x) - target_pressure

            res = root(fun, x0=[0.1], method='lm')
            if not res.success:
                raise RuntimeError(f'Root finding failed for pressure {target_pressure}: {res.message}')
            return res.x.item()

        loading = np.array([solve_single(target_pressure)
                            for target_pressure in pressure_values])
        if scalar_input:
            return loading.item()
        return loading.reshape(pressure_array.shape)


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
        pass

    def initial_guess(self):
        """Provides initial guess for model parameters.

        For the Dual Site Langmuir isotherm, we follow the scheme of pyIAST and assume
        parameter values based on the Langmuir model.
        """
        langmuir_guess = super().initial_guess()
        return {
            'M': langmuir_guess['M'],
            'K': langmuir_guess['K'],
            't10': 0.0,
        }
