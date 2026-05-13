"""

"""

import numpy as np

from pyrast.isotherms.model_isotherm import ModelIsotherm


class TemkinApprox(ModelIsotherm, model_name='TemkinApprox'):

    # Class variables for every instance
    name = 'TemkinApprox'
    param_names = ('M', 'K', 'theta')
    param_default_bounds = ((0., np.inf), (0., np.inf), (0., np.inf))

    def loading(self, pressure: float):
        langmuir_frac_loading = self.model_parameters['K'] * pressure / \
                                (1.0 + self.model_parameters['K'] * pressure)
        return self.model_parameters['M'] * \
                (langmuir_frac_loading + self.model_parameters['theta'] * \
                 langmuir_frac_loading**2 * (langmuir_frac_loading - 1.0))

    def spreading_pressure(self, pressure: float):
        return (self.model_parameters["M"] *
                np.log(1.0 + self.model_parameters["K"] * pressure))

    def initial_guess(self):
        langmuir_guess = super().initial_guess()
        return {'M': langmuir_guess['M'],
                'K': langmuir_guess['K'],
                'theta': 0.0}
