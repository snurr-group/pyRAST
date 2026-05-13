"""

"""

import numpy as np

from pyrast.isotherms.model_isotherm import ModelIsotherm


class DSLangmuir(ModelIsotherm, model_name='DSLangmuir'):

    # Class variables for every instance
    name = 'DSLangmuir'
    param_names = ('M1', 'K1', 'M2', 'K2')
    param_default_bounds = ((0., np.inf), (0., np.inf), (0., np.inf), (0., np.inf))

    def loading(self, pressure: float):
        k1p = self.model_parameters["K1"] * pressure
        k2p = self.model_parameters["K2"] * pressure
        return self.model_parameters["M1"] * k1p / (1.0 + k1p) + \
               self.model_parameters["M2"] * k2p / (1.0 + k2p)

    def spreading_pressure(self, pressure: float):
        return self.model_parameters["M1"] * np.log(
            1.0 + self.model_parameters["K1"] * pressure) +\
               self.model_parameters["M2"] * np.log(
                   1.0 + self.model_parameters["K2"] * pressure)

    def initial_guess(self):
        langmuir_guess = super().initial_guess()
        return {
            "M1": 0.5 * langmuir_guess['M'],
            "K1": 0.4 * langmuir_guess['K'],
            "M2": 0.5 * langmuir_guess['M'],
            "K2": 0.6 * langmuir_guess['K'],
        }
