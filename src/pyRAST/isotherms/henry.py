"""

"""

import numpy as np

from pyrast.isotherms.model_isotherm import ModelIsotherm


class Henry(ModelIsotherm, model_name='Henry'):

    # Class variables for every instance
    name = 'Henry'
    param_names = ('KH',)
    param_default_bounds = ((0., np.inf),)

    def loading(self, pressure: float):
        return self.model_parameters['KH'] * pressure

    def spreading_pressure(self, pressure: float):
        return self.model_parameters['KH'] * pressure

    def initial_guess(self):
        langmuir_guess = super().initial_guess()
        return {'KH': langmuir_guess['M'] * langmuir_guess['K']}
