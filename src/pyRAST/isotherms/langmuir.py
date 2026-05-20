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
        return (self.model_parameters["M"] *
                np.log(1.0 + self.model_parameters["K"] * pressure))

    def pressure(self, spreading_pressure: float):
        return (1.0 / self.model_parameters["K"]) * \
                (np.exp(spreading_pressure / self.model_parameters["M"]) - 1.0)

    def initial_guess(self):
        return super().initial_guess()
