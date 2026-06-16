"""Wilson Vacancy Solution Theory isotherm model."""

import numpy as np
from scipy.optimize import root

from pyrast.isotherms.model_isotherm import ModelIsotherm


class WVST(ModelIsotherm, model_name='W-VST'):

    # Class variables for every instance
    name = 'W-VST'
    param_names = ('M', 'K', 'L1v', 'Lv1')
    param_default_bounds = ((0., np.inf), (0., np.inf), (-np.inf, np.inf),
                            (-np.inf, np.inf))

    def pressure(self, loading):
        r"""UPDATE"""
        m = self.model_parameters['M']
        k = self.model_parameters['K']
        l1v = self.model_parameters['L1v']
        lv1 = self.model_parameters['Lv1']

        cov = loading / m
        langmuir = m * cov / (k * (1.0 - cov))
        wilson1 = l1v * (1.0 - (1.0 - l1v) * cov) / (l1v + (1.0 - l1v) * cov)
        wilson2 = np.exp(- (lv1 * (1.0 - lv1) * cov)/(1.0 - (1.0 - lv1) * cov) -
                         ((1.0 - l1v) * cov)/(l1v + (1.0 - l1v) * cov))
        return langmuir * wilson1 * wilson2

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
            'L1v': 1.0,
            'Lv1': 1.0,
        }
