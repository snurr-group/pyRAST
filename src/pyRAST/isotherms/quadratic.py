"""

"""

import numpy as np

from pyrast.isotherms.model_isotherm import ModelIsotherm


class Quadratic(ModelIsotherm, model_name='Quadratic'):

    # Class variables for every instance
    name = 'Quadratic'
    param_names = ('M', 'Ka', 'Kb')
    param_default_bounds = ((0., np.inf), (0., np.inf), (0., np.inf))

    def loading(self, pressure: float):
        return self.model_parameters['M'] * \
        (self.model_parameters['Ka'] + 2.0 * self.model_parameters['Kb'] * pressure) * \
        pressure / (1.0 + self.model_parameters['Ka'] * pressure +
                     self.model_parameters['Kb'] * pressure**2)

    def spreading_pressure(self, pressure: float):
        return self.model_parameters['M'] * \
                np.log(1.0 + self.model_parameters['Ka'] * pressure +
                        self.model_parameters['Kb'] * pressure**2)

    def initial_guess(self):
        langmuir_guess = super().initial_guess()
        return {'M': langmuir_guess['M'] / 2.0,
                'Ka': langmuir_guess['K'],
                'Kb': langmuir_guess['K']**2.0}
