"""

"""

import numpy as np

from pyrast.isotherms.model_isotherm import ModelIsotherm


class Langmuir(ModelIsotherm, model_name='Langmuir'):

    # Class variables for every instance
    name = 'Langmuir'
    param_names = ('M', 'K')
    param_default_bounds = ((0., np.inf), (0., np.inf))

    def loading(self, pressure: float):
        return self.model_parameters['M'] * self.model_parameters['K'] * pressure / \
                (1.0 + self.model_parameters['K'] * pressure)

    def spreading_pressure(self, pressure: float):
        return (self.model_parameters['M'] *
                np.log(1.0 + self.model_parameters['K'] * pressure))

    def p0(self, target_phi: float):
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
        return super().initial_guess()
