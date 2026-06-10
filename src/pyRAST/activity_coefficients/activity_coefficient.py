"""Base model for activity coefficient models."""

from typing import cast

import numpy as np
from scipy.optimize import brentq

from pyrast.isotherms.interpolator_isotherm import CubicIsotherm, InterpolatorIsotherm


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
    model: str
    c: float
    gamma_tol: float
    param_tol: float
    max_iter: int

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
                 c: float = 1, param_tol: float = 1e-4, gamma_tol: float = 1e-4,
                 max_iter: int = 100, param_mixing: float = 0.2,
                 verbose: bool = False,
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
            self.c = c
        # Multiple fugacity points provided as 2D arrays
        else:
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

        # Store calculation info
        self.param_tol = param_tol
        self.gamma_tol = gamma_tol
        self.max_iter = max_iter

        # Fit model to data
        # If user provided parameters, check that keys are correct
        if model_parameters is not None:
            if set(model_parameters.keys()) != set(self.param_names):
                raise ValueError(f'model_parameters keys must be {self.param_names}.')
            self.model_parameters = model_parameters
        else:
            self.model_parameters = dict.fromkeys(self.param_names, np.nan)
            if assume_ideal_gamma:
                self._fit_to_gamma(verbose=verbose)
            else:
                self._rigorous_fit_to_gamma(param_mixing, verbose=verbose)


    def __repr__(self):
        """String representation of activity coefficient model."""
        return (f'{self.name} activity coefficient model with parameters: '
                f'{self.model_parameters}')

    def ln_gamma(self, x, phi):
        """Calculates natural log of activity coefficients."""
        raise NotImplementedError('ln_gamma method must be implemented in subclass.')

    def gamma(self, x, phi):
        """Calculates activity coefficients (gamma) from ln_gamma."""
        return np.exp(self.ln_gamma(x, phi))

    def inverse_excess_loading(self, x, phi):
        """Calculates inverse excess loading."""
        raise NotImplementedError('inverse_excess_loading method must be implemented '
                                  'in subclass.')

    def _gamma_from_loadings(self, comp_q, y, total_f, *, excess_loading = False,
                             verbose: bool = False):
        """docstring"""
        # Calculate important variables for determining gamma and phi
        q_total = sum(comp_q)
        x = comp_q / q_total
        p0 = np.zeros(len(self.isotherms))

        # Define residual function for root finding to solve for phi
        def residuals(phi, q_excess = 0.0):
            for i in range(len(self.isotherms)):
                p0[i] = self.isotherms[i].p0(phi)
            q0 = np.array([self.isotherms[i].loading(p0[i])
                           for i in range(len(self.isotherms))])
            q_total_pred = 1.0 / (np.sum(x / q0) + q_excess)
            return q_total_pred - q_total

        # Use a bracketing strategy to ensure brentq can find a root.
        def _bracket_phi(residuals, q_excess = 0.0, phi_low = 1e-12, phi_high = 1.0,
                         max_expand = 60, phi_cap = np.inf):
            # Determine the maximum phi based on isotherm extrapolation limits
            def _phi_cap_for_isotherm(iso, total_f):
                if isinstance(iso, InterpolatorIsotherm):
                    if iso.extrap_method is not None or iso.fill_value is not None:
                        return iso.extrap_p
                    p_max = iso.df[iso.pressure_key].max()
                    return iso.spreading_pressure(p_max)
                if isinstance(iso, CubicIsotherm):
                    if iso.extrap_method is not None:
                        return iso.extrap_p
                    p_max = iso.df[iso.pressure_key].max()
                    return iso.spreading_pressure(p_max)
                return np.inf

            # Compute the cap if there is one
            caps = [_phi_cap_for_isotherm(iso, total_f) for iso in self.isotherms]
            phi_cap = min(caps) if any(np.isfinite(caps)) else np.inf

            # Check lower limit to ensure we don't start with an invalid point
            f_low = residuals(phi_low, q_excess=q_excess)
            if not np.isfinite(f_low):
                raise ValueError('Residual is not finite at initial phi_low.')

            # Lower upper limit if necessary to find a finite residual
            f_high = residuals(phi_high, q_excess=q_excess)
            while not np.isfinite(f_high) and phi_high > phi_low:
                phi_high *= 0.9
                f_high = residuals(phi_high, q_excess=q_excess)

            # Throw error if we can't find a finite residual at the upper limit
            if not np.isfinite(f_high):
                raise ValueError('Could not find finite residual at initial phi_high.')

            # Expand upper limit until we bracket a root or hit the cap
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

            # If we exit the loop without finding a valid bracket, raise an error
            raise ValueError("Could not find valid bracketing for root finding")

        # Solve for phi without excess loading correction with bracketed root finding
        phi_low, phi_high = _bracket_phi(residuals)
        phi_sol = cast('float', brentq(residuals, phi_low, phi_high))

        # Calculate gamma from loadings and phi without excess loading correction
        for i in range(len(self.isotherms)):
            p0[i] = self.isotherms[i].p0(phi_sol)
        gamma = (y * total_f) / (p0 * x)
        if not excess_loading:
            return gamma, phi_sol

        # Iterative excess loading correction
        for iteration in range(self.max_iter):
            # Copy old gamma
            gamma_old = gamma.copy()
            q_excess = self.inverse_excess_loading(x, phi_sol)

            # Resolve phi with excess loading correction
            phi_low, phi_high = _bracket_phi(residuals, q_excess = q_excess)
            phi_sol = cast('float', brentq(residuals,
                                           phi_low, phi_high, args=(q_excess,)))

            # Calculate new gamma
            for i in range(len(self.isotherms)):
                p0[i] = self.isotherms[i].p0(phi_sol)
            gamma = (y * total_f) / (p0 * x)

            # Print convergence info if verbose
            if verbose:
                print(f'Excess loading correction loop: iteration {iteration + 1}, '
                      f'phi: {phi_sol}, gamma: {gamma}')

            # Check convergence
            if np.all(np.abs(gamma - gamma_old) < self.gamma_tol):
                if verbose:
                    print(f'Gamma converged after {iteration + 1} iterations.')
                break
        # If we reach max iterations without convergence, print warning
        else:
            print('Gamma from loadings did not converge.')

        return gamma, phi_sol

    def _fit_to_gamma(self, *, excess_loading = False, verbose: bool = False):
        """docstring"""
        return NotImplementedError('_fit_to_gamma method must be implemented in'
                                   'subclass.')

    def _rigorous_fit_to_gamma(self, param_mixing: float, *, verbose: bool = False):
        """docstring"""
        # First pass for model parameters is use ideal case
        self._fit_to_gamma(verbose=verbose)

        # Now we want to iteratively get more accurate model parameters
        for iteration in range(self.max_iter):
            # Copy old parameters
            params_old = self.model_parameters.copy()
            self._fit_to_gamma(excess_loading=True, verbose=verbose)

            # Mix new and old parameters to improve stability
            params_new = self.model_parameters.copy()
            for k in self.param_names:
                self.model_parameters[k] = param_mixing * params_new[k] + \
                    (1 - param_mixing) * params_old[k]

            # Print parameters at each iteration if verbose
            if verbose:
                print(f'Model parameter convergence loop: iteration {iteration + 1}, '
                      f'model parameters: {self.model_parameters}')

            # Check convergence
            if all(abs(self.model_parameters[param] - params_old[param]) <
                   self.param_tol for param in self.param_names):
                if verbose:
                    print(f'Model parameters converged after {iteration + 1} '
                          'iterations.')
                break

        # If we reach max iterations without convergence, print warning
        else:
            print('Model parameters did not converge.')
