"""
functions of activity coefficient parent class go here

"""
# ruff: noqa: TC002

from typing import cast

import numpy as np
from scipy.optimize import brentq


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
                 comp_q: np.ndarray | list, isotherms: list, model: str,
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
            self._fit_to_gamma()


    def __repr__(self):
        return (f'{self.name} activity coefficient model with parameters: '
                f'{self.model_parameters}')

    def ln_gamma(self, x, phi):
        """docstring"""
        raise NotImplementedError('ln_gamma method must be implemented in subclass.')

    def gamma(self, x, phi):
        """docstring"""
        return np.exp(self.ln_gamma(x, phi))

    def inverse_excess_loading(self, x):
        """docstring"""
        raise NotImplementedError('inverse_excess_loading method must be implemented in subclass.')

    def _gamma_from_loadings(self, comp_q, y, total_f):
        """docstring"""
        q_total = sum(comp_q)
        x = comp_q / q_total
        p0 = np.zeros(len(self.isotherms))

        def residuals(phi):
            for i in range(len(self.isotherms)):
                p0[i] = self.isotherms[i].pressure(phi)
            q0 = np.array([self.isotherms[i].loading(p0[i])
                           for i in range(len(self.isotherms))])
            # We make an assumption of ideality here, may need to change later
            q_total_pred = 1.0 / np.sum(x / q0)
            return q_total_pred - q_total

        phi_max = max(iso.spreading_pressure(total_f * 10)
                      for iso in self.isotherms)
        phi_sol = cast('float', brentq(residuals, 1e-10, phi_max))

        for i in range(len(self.isotherms)):
            p0[i] = self.isotherms[i].pressure(phi_sol)
        gamma = (y * total_f) / (p0 * x)

        return gamma, phi_sol

    def _fit_to_gamma(self):
        """docstring"""
        '''gamma, phi = self._gamma_from_loadings(self.comp_q, self.y, self.total_f)
        x = self.comp_q / np.sum(self.comp_q)

        def residuals(params):
            self.model_parameters = dict(zip(self.param_names, params))
            return self.ln_gamma(x, phi) - np.log(gamma)

        res = fsolve(residuals, [1.0] * len(self.param_names))

        self.model_parameters = dict(zip(self.param_names, res))'''
        pass


