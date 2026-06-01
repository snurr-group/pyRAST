"""
functions of activity coefficient parent class go here

"""

from typing import cast

import numpy as np
from scipy.optimize import brentq

from pyrast.isotherms.interpolator_isotherm import InterpolatorIsotherm


class ActivityCoefficient:
    # Model list built at import time
    _MODELS = {}

    # Class variables
    name: str = ''
    param_names: tuple
    param_default_bounds: tuple

    # Instance variables
    total_f: float | np.ndarray
    y: np.ndarray
    comp_q: np.ndarray
    isotherms: list
    model_parameters: dict
    model: str | None = None

    def __new__(cls, model: str = '', *args, **kwargs):
        """docstring"""
        if cls is ActivityCoefficient:
            if model not in cls._MODELS:
                raise ValueError(f'{model} is not a valid model. Choose from'
                                 f' {list(cls._MODELS.keys())}')
            subclass = cls._MODELS[model]
            return super().__new__(subclass)
        return super().__new__(cls)

    def __init_subclass__(cls, model_name: str = '', *args, **kwargs):
        """docstring"""
        super().__init_subclass__(**kwargs)
        if model_name:
            ActivityCoefficient._MODELS[model_name] = cls

    def __init__(self, total_f: np.ndarray | list | float, y: np.ndarray | list,
                 comp_q: np.ndarray | list, isotherms: list, model: str, *,
                 assume_ideal_gamma: bool = False,
                 model_parameters: dict | None = None):
        """ One line description

        Args:
            param1(type): Description of param1

        Returns:
            type: Description of return value

        """
        # If total_f is a float, ensure y and comp_q are 1D
        if isinstance(total_f, (int, float)):
            total_f = float(total_f)
            if total_f <= 0:
                raise ValueError('Total fugacity must be positive.')
            if len(y) != len(comp_q):
                raise ValueError('Length of y and comp_q must be the same.')
            if not np.isclose(sum(y), 1.0):
                raise ValueError('Gas phase mole fractions must sum to 1.0.')
            if not all(q >= 0 for q in comp_q):
                raise ValueError('Adsorbed phase loadings must be non-negative.')
            if len(isotherms) != len(y):
                raise ValueError('Length of isotherms must match length of y and comp_q'
                                 '.')
        else: # Multiple fugacity points provided as 2D arrays
            total_f = np.asarray(total_f, dtype=float)
            for f in total_f:
                if f <= 0:
                    raise ValueError('Total fugacity must be positive.')
            y = np.asarray(y)
            comp_q = np.asarray(comp_q)
            if y.shape != comp_q.shape:
                raise ValueError('y and comp_q must have the same dimensions.')
            if not np.allclose(np.sum(y, axis=1), 1.0):
                raise ValueError('All gas phase mole fractions must sum to 1.0.')
            if not np.all(comp_q >= 0):
                raise ValueError('All adsorbed phase loadings must be non-negative.')
            if len(isotherms) != y.shape[1]:
                raise ValueError('Length of isotherms must match number of components '
                                 'in y and comp_q.')

        # Store data
        self.total_f = total_f
        self.y = np.asarray(y)
        self.comp_q = np.asarray(comp_q)
        self.isotherms = isotherms

        # Store model info
        self.model = model

        # Fit model to data
        # If user provided parameters, check that keys are correct
        if model_parameters is not None:
            if set(model_parameters.keys()) != set(self.param_names):
                raise ValueError(f'model_parameters keys must be {self.param_names}.')
            self.model_parameters = model_parameters
        else:
            self.model_parameters = dict.fromkeys(self.param_names, np.nan)
            if assume_ideal_gamma:
                self._fit_to_gamma()
            else:
                self._rigorous_fit_to_gamma()


    def __repr__(self):
        return (f'{self.name} activity coefficient model with parameters: '
                f'{self.model_parameters}')

    def ln_gamma(self, x, phi):
        """docstring"""
        raise NotImplementedError('ln_gamma method must be implemented in subclass.')

    def gamma(self, x, phi):
        """docstring"""
        return np.exp(self.ln_gamma(x, phi))

    def inverse_excess_loading(self, x, phi):
        """docstring"""
        raise NotImplementedError('inverse_excess_loading method must be implemented '
                                  'in subclass.')

    def _gamma_from_loadings(self, comp_q, y, total_f, *, excess_loading = False,
                             max_iter = 100, tol = 1e-3):
        """docstring"""
        q_total = sum(comp_q)
        x = comp_q / q_total
        p0 = np.zeros(len(self.isotherms))

        def residuals(phi, q_excess = 0.0):
            for i in range(len(self.isotherms)):
                p0[i] = self.isotherms[i].pressure(phi)
            q0 = np.array([self.isotherms[i].loading(p0[i])
                           for i in range(len(self.isotherms))])
            # We make an assumption of ideality here, may need to change later
            q_total_pred = 1.0 / (np.sum(x / q0) + q_excess)
            return q_total_pred - q_total

        # Use a bracketing strategy to ensure brentq can find a root.
        def _bracket_phi(residuals, q_excess = 0.0, phi_low = 1e-12, phi_high = 1.0,
                         max_expand = 60, phi_cap = np.inf):
            def _phi_cap_for_isotherm(iso, total_f):
                if isinstance(iso, InterpolatorIsotherm):
                    if iso.extrap_method is not None or iso.fill_value is not None:
                        return np.inf
                    p_max = iso.df[iso.pressure_key].max()
                    return iso.spreading_pressure(p_max)
                return np.inf

            # Compute the cap if there is one
            caps = [_phi_cap_for_isotherm(iso, total_f) for iso in self.isotherms]
            phi_cap = min(caps) if any(np.isfinite(caps)) else np.inf

            f_low = residuals(phi_low, q_excess=q_excess)
            if not np.isfinite(f_low):
                raise ValueError('Residual is not finite at initial phi_low.')

            f_high = residuals(phi_high, q_excess=q_excess)
            while not np.isfinite(f_high) and phi_high > phi_low:
                phi_high *= 0.5
                f_high = residuals(phi_high, q_excess=q_excess)

            if not np.isfinite(f_high):
                raise ValueError('Could not find finite residual at initial phi_high.')

            for _ in range(max_expand):
                if np.isfinite(f_high) and np.sign(f_low) != np.sign(f_high):
                    return phi_low, phi_high

                if np.isfinite(phi_cap) and phi_high >= phi_cap:
                    break

                phi_high = min(phi_high * 2.0,
                               phi_cap) if np.isfinite(phi_cap) else phi_high * 2.0
                f_high = residuals(phi_high, q_excess=q_excess)

                # if we just hit a NaN/inf, back off once
                if not np.isfinite(f_high):
                    phi_high *= 0.5
                    f_high = residuals(phi_high, q_excess=q_excess)
                    if not np.isfinite(f_high):
                        break

            raise ValueError("Could not find valid bracketing for root finding")

        phi_low, phi_high = _bracket_phi(residuals)
        # phi_max = max(iso.spreading_pressure(total_f * 10)
        #               for iso in self.isotherms)
        phi_sol = cast('float', brentq(residuals, phi_low, phi_high))

        for i in range(len(self.isotherms)):
            p0[i] = self.isotherms[i].pressure(phi_sol)
        gamma = (y * total_f) / (p0 * x)
        if not excess_loading:
            return gamma, phi_sol

        # Iterative excess loading correction
        for iteration in range(max_iter):
            gamma_old = gamma.copy()
            q_excess = self.inverse_excess_loading(x, phi_sol)

            phi_low, phi_high = _bracket_phi(residuals, q_excess = q_excess)
            # Resolve phi with excess loading correction
            phi_sol = cast('float', brentq(residuals,
                                           phi_low, phi_high, args=(q_excess,)))
            for i in range(len(self.isotherms)):
                p0[i] = self.isotherms[i].pressure(phi_sol)
            gamma = (y * total_f) / (p0 * x)

            if np.all(np.abs(gamma - gamma_old) < tol):
                break
        else:
            print('gamma from loadings did not converge')

        return gamma, phi_sol

    def _fit_to_gamma(self, *, excess_loading = False):
        """docstring"""
        '''gamma, phi = self._gamma_from_loadings(self.comp_q, self.y, self.total_f)
        x = self.comp_q / np.sum(self.comp_q)

        def residuals(params):
            self.model_parameters = dict(zip(self.param_names, params))
            return self.ln_gamma(x, phi) - np.log(gamma)

        res = fsolve(residuals, [1.0] * len(self.param_names))

        self.model_parameters = dict(zip(self.param_names, res))'''
        pass

    def _rigorous_fit_to_gamma(self, max_iter = 100, tol = 1e-6):
        """docstring"""
        # First pass for model parameters is use ideal case
        self._fit_to_gamma()

        # Now we want to iteratively get more accurate model parameters
        for iteration in range(max_iter):
            params_old = self.model_parameters.copy()
            self._fit_to_gamma(excess_loading=True)

            alpha = 0.2
            params_new = self.model_parameters.copy()

            for k in self.param_names:
                self.model_parameters[k] = alpha * params_new[k] + (1 - alpha) * \
                    params_old[k]

            # Check convergence
            print(f'Iteration {iteration + 1}, model parameters: {self.model_parameters}')
            if all(abs(self.model_parameters[param] - params_old[param]) < tol
                   for param in self.param_names):
                print(f'Converged after {iteration + 1} iterations.')
                break

        # If we reach max iterations without convergence, print warning
        else:
            print('did not converge')
