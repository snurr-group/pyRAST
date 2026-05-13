"""

"""

import numpy as np

from pyrast.isotherms.model_isotherm import ModelIsotherm


class BET(ModelIsotherm, model_name='BET'):

    # Class variables for every instance
    name = 'BET'
    param_names = ('M', 'Ka', 'Kb')
    param_default_bounds = ((0., np.inf), (0., np.inf), (0., np.inf))

    def loading(self, pressure: float):
        return self.model_parameters['M'] * self.model_parameters['Ka'] * pressure / (
                (1.0 - self.model_parameters['Kb'] * pressure) *
                (1.0 - self.model_parameters['Kb'] * pressure +
                 self.model_parameters['Ka'] * pressure))

    def spreading_pressure(self, pressure: float):
        return self.model_parameters["M"] * np.log(
            (1.0 - self.model_parameters['Kb'] * pressure +
             self.model_parameters['Ka'] * pressure) /
            (1.0 - self.model_parameters['Kb'] * pressure))

    def initial_guess(self):
        langmuir_guess = super().initial_guess()
        return {'M': langmuir_guess['M'],
                'Ka': langmuir_guess['K'],
                'Kb': langmuir_guess['K'] * 0.01}
