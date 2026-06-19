"""Adsorption NRTL Vacancy Solution Theory isotherm model."""

import numpy as np

from pyrast.isotherms.model_isotherm import ModelIsotherm


class ANRTLVST(ModelIsotherm, model_name='aNRTL-VST'):

    # Class variables for every instance
    name = 'aNRTL-VST'
    param_names = ('M', 'K', 't10')
    param_default_bounds = ((0., np.inf), (0., np.inf), (-np.inf, np.inf))

    interp_load = None
    interp_spread = None
    interp_p0 = None

    def pressure(self, loading):
        r"""Calculates pressure as a function of loading.

        Vacancy Solution Theory (VST) models are defined as functions of pressure.
        Unfortunately, there is no analytical function for loading, spreading pressure,
        or p0 as a result. This function is used in combination with root solving to
        support fitting the VST isotherm with the Wilson activity coefficient model. The
        Adsorption NRTL VST isotherm has three parameters, two from the Langmuir
        isotherm and one from the activity coefficient model.

        Pressure in the Adsorption NRTL VST isotherm is given as:

        .. math::

            P(q) = \frac{M}{K} \frac{\theta}{1-\theta} FINISH THIS

        Args:
            Loading(float or np.ndarray): loadings(s) at which to calculate pressure

        Returns:
            float or np.ndarray: pressure as same variable type as input
        """
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
        """Returns loading as a function of pressure (or fugacity).

        As vacancy solution models have implicit functions for loading, we must use
        interpolated functions for loading, spreading pressure, and p0. Interpolants
        are built after fitting the models. Here, loading is calculated as a
        function of pressure using an interpolator, if built, or it defaults to the
        ModelIsotherm parent class, which uses root solving to determine loading
        from the pressure function specific to VST.

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate loading

        Returns:
            float or np.ndarray: loading as same variable type as input
        """
        if self.interp_load is not None:
            return self.interp_load(pressure)
        return super().loading(pressure)

    def spreading_pressure(self, pressure):
        """Returns spreading pressure as a function of pressure (or fugacity).

        As vacancy solution models have implicit functions for loading, we must use
        interpolated functions for loading, spreading pressure, and p0. Interpolants
        are built after fitting the models. Here, spreading pressure is calculated as a
        function of phi using an interpolator, if built, or it defaults to the
        ModelIsotherm parent class, which will raise an exception. The spreading
        pressure interpolator should always be built after model fitting.

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate spreading
                pressure

        Returns:
            float or np.ndarray: spreading pressure as same variable type as input
        """
        if self.interp_spread is not None:
            return self.interp_spread(pressure)
        # Fallback to the parent class method if no interpolation is available
        return super().spreading_pressure(pressure)

    def p0(self, phi):
        """Returns P0 as a function of spreading pressure.

        As vacancy solution models have implicit functions for loading, we must use
        interpolated functions for loading, spreading pressure, and p0. Interpolants
        are built after fitting the models. Here, p0 is calculated as a function of phi
        using an interpolator, if built, or it defaults to the ModelIsotherm parent
        class, which uses root solving.

        Args:
            target_phi (float): Spreading pressure to calculate P0

        Returns:
            float: P0 value
        """
        if self.interp_p0 is not None:
            return self.interp_p0(phi)
        # Fallback to the parent class method if no interpolation is available
        return super().p0(phi)

    def initial_guess(self):
        """Provides initial guess for model parameters.

        For the Adsorption NRTL Vacancy Solution Theory isotherm, we assume the case of
        a Langmuir isotherm with activity coefficient of 1.
        """
        langmuir_guess = super().initial_guess()
        return {
            'M': langmuir_guess['M'],
            'K': langmuir_guess['K'],
            't10': 0.0,
        }
