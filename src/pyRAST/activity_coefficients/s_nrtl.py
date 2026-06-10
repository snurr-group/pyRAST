"""Implementation of Symmetric NRTL Model"""

import numpy as np
from scipy.optimize import least_squares, root

from pyrast.activity_coefficients.activity_coefficient import ActivityCoefficient


class SNRTL(ActivityCoefficient, model_name='sNRTL'):
    r"""
    The Symmetric NRTL model is analagous to the Symmetric NRTL model for vapor liquid
    equlibria. The Symmetric NRTL model is best suited for: UPDATE

    The excess Gibbs free energy in the Symmetric NRTL model is given by:

    .. math::
        \frac{g^E}{RT} = \frac{x_1 x_2 \tau_{12} (G_{12} - 1)}
            {x_1 G_{12} + x_2} (1 - e^{-C \phi})

        G_{12} = \exp(-\alpha \tau_{12}), \ \tau_{12} = \tau_{21}, \ \alpha = 0.3

    Sources: Kaur, H., Tun, H., Sees, M. & Chen, C.-C. Local composition activity
    coefficient model for mixed-gas adsorption equilibria. Adsorption 25, 951-964
    (2019).

    Kopatsis, A., Salinger, A. & Myers, A. L. Thermodynamics of solutions with solvent
    and solute in different pure states. AIChE Journal 34, 1275-1286 (1988).
    """
    # Class variables for every instance
    name = 'sNRTL'
    param_names = ('t12', 'C')
    param_default_bounds = ((-np.inf, np.inf), (0.0, np.inf))
    alpha = 0.3

    def ln_gamma(self, x, phi):
        r"""Calculates the natural log of the activity coefficients for each component.

        In the Symmetric NRTL model, the activity coefficients are calculated as:

        .. math::
            \ln \gamma_1 = \frac{x_2^2 \tau_{12} (G_{12} - 1)}{(x_1 G_{12} + x_2)^2}
            (1 - e^{-C \phi})

            \ln \gamma_2 = \frac{x_1^2 \tau_{21} (G_{21} - 1)}{(x_2 G_{21} + x_1)^2}
            (1 - e^{-C \phi})

        Args:
            x (array-like): Mole fractions of the components in the mixture.
            phi (float): Spreading pressure for the mixture.
        Returns:
            np.ndarray: Natural log of the activity coefficients for each component.
        """
        t12 = self.model_parameters['t12']
        t21 = t12
        c = self.model_parameters['C']
        f = 1.0 - np.exp(-c * phi)
        alpha = self.alpha
        g12 = np.exp(-alpha*t12)
        g21 = np.exp(-alpha*t21)

        ln_gamma0 = x[1]**2 * f * t12 * (g12 - 1.0) / (x[0] * g12 + x[1])**2
        ln_gamma1 = x[0]**2 * f * t21 * (g21 - 1.0) / (x[1] * g21 + x[0])**2
        return np.array([ln_gamma0, ln_gamma1])

    def inverse_excess_loading(self, x, phi):
        r"""Calculates the inverse of the excess loading given composition and phi.

        The excess loading in the Symmetric NRTL model is calculated as:

        .. math:: \left(\frac{1}{q}\right)^E = \frac{x_1 x_2 \tau_{12} (G_{12} - 1)}
            {x_1 G_{12} + x_2} C e^{-C \phi}

        Args:
            x (array-like): Mole fractions of the components in the mixture.
            phi (float): Spreading pressure for the mixture.
        Returns:
            float: Inverse of the excess loading for the mixture.
        """
        t12 = self.model_parameters['t12']
        g12 = np.exp(-self.alpha*t12)
        c = self.model_parameters['C']

        return c * x[0] * x[1] * t12 * (g12 - 1.0) * np.exp(-c * phi) / \
               (x[0]*g12 + x[1])

    def _fit_component_loadings(self, *, excess_loading: bool = False,
                                verbose: bool = False):
        """docstring"""
        if self.loadings.shape[0] == 1:
            # Handle the case where a single data point is provided, thus c is assumed
            gamma, phi = self._gamma_from_loadings(self.loadings, self.partial_fug,
                                                   excess_loading=excess_loading,
                                                   verbose=verbose)
            x = self.loadings / np.sum(self.loadings)
            c = self.c
            ln_g = np.log(gamma)

            def equations(p):
                t12 = p[0]
                self.model_parameters = {'t12': t12, 'C': c}
                ln_gamma = self.ln_gamma(x, phi)
                return ln_gamma - ln_g

            sol = root(equations, x0=[1.0], method='hybr', tol=1e-10)

            if not sol.success:
                raise ValueError(
                    f"sNRTL parameter fit failed: {sol.message}. "
                    "Try a different initial guess or check data quality.",
                )

            t12 = sol.x[0]
            self.model_parameters = {'t12': t12, 'C': c}
        else:
            # Handle the case where multiple data points are provided
            # In this case, we fit all parameters simultaneously using least squares
            points = len(self.partial_fug)
            gamma = np.zeros((points, 2))
            phi = np.zeros(points)
            xs = np.zeros((points, 2))

            for i in range(points):
                gamma[i], phi[i] = self._gamma_from_loadings(self.loadings[i],
                                                             self.partial_fug[i],
                                                          excess_loading=excess_loading,
                                                             verbose=verbose)
                xs[i] = self.loadings[i] / np.sum(self.loadings[i])

            def residuals(params):
                t12 = params[0]
                c = params[1]
                self.model_parameters = {'t12': t12, 'C': c}

                res = np.zeros(points * 2)
                for i in range(points):
                    ln_gamma_pred = self.ln_gamma(xs[i], phi[i])
                    ln_gamma_exp = np.log(gamma[i])
                    res[2*i:2*i+2] = ln_gamma_pred - ln_gamma_exp
                return res

            res = least_squares(residuals, x0=[1.0, 0.5], xtol=self.param_tol,
                                ftol=self.param_tol)
            if not res.success:
                raise ValueError(
                    f"sNRTL parameter fit failed: {res.message}. "
                    "Try a different initial guess or check data quality.",
                )

            t12, c = res.x[0], res.x[1]
            self.model_parameters = {'t12': t12, 'C': c}

