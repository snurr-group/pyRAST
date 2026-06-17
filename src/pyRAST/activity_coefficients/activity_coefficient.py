"""Base model for activity coefficient models."""

from typing import cast

import numpy as np
from scipy.optimize import brentq, least_squares

from pyrast.isotherms.interpolator_isotherm import CubicIsotherm, InterpolatorIsotherm


class ActivityCoefficient:
    # Model list built at import time
    _MODELS = {}

    # Class variables
    name: str = ''
    param_names: tuple
    param_default_bounds: tuple
    param_ideal_values: tuple

    # Instance variables
    partial_fug: np.ndarray
    loadings: np.ndarray
    isotherms: list
    model_parameters: dict
    model: str

    def __new__(cls, *args, **kwargs):
        """Creates an instance of the user-specified model.

        This factory design pattern is identical to the model isotherms and allows
        users to easily switch between different activity coefficient models. Users
        should never interact with this method directly.
        """
        if cls is ActivityCoefficient:
            model = kwargs.pop('model', None)
            if model is None:
                if args and isinstance(args[0], str):
                    model = args[0]
                elif len(args) >= 4:
                    model = args[3]
            if model not in cls._MODELS:
                raise ValueError(f'{model} is not a valid model. Choose from'
                                 f' {list(cls._MODELS.keys())}')
            subclass = cls._MODELS[model]
            return super().__new__(subclass)
        return super().__new__(cls)

    def __init_subclass__(cls, *, model_name: str = '', **kwargs):
        """Registers subclasses of ModelIsotherm at import time.

        Users should never interact with this method directly. To modify the list of
        available models, edit the import statements in __init__.py.
        """
        super().__init_subclass__(**kwargs)
        if model_name:
            ActivityCoefficient._MODELS[model_name] = cls

    def __init__(self, partial_fug: np.ndarray | list, loadings: np.ndarray | list,
                 isotherms: list, model: str, *, total_loading: bool = False,
                 verbose: bool = False, model_parameters: dict | None = None,
                 c: float = 1, param_guess: dict | None = None,
                 param_bounds: dict | None = None,
                 optimization_options: dict | None = None,
                 param_tol: float = 1e-6):
        """Initializes an activity coefficient model.

        Activity coefficient models are used in real adsorbed solution theory (RAST) to
        account for non-ideal interactions between adsorbed species. This base class
        provides the framework for fitting activity coefficient models from binary
        mixture data. There are two ways to fit the model parameters:

        1) Component loadings: 

        2) Total loading: If the user provides total loadings for different partial
        fugacities, the model parameters will be fit by minimizing the residual
        between the predicted total loading from a RAST calculation and the provided
        total loading. This approach is more computationally intensive, but is
        useful for experimental data where component loadings are difficult to
        measure. The user must provide at least as many total loading data points as
        model parameters.

        For a full discussion of the fitting procedure and the underlying equations,
        see the documentation or paper discussion. Fitting activity coefficient
        models can be challenging, and convergence depends on the quality of the data
        and the choice of fitting options. If you have trouble fitting a model, try
        adjusting the initial guess, tolerances, or optimization options.

        Args:
            partial_fug (array-like): Partial fugacities of each component in
                the gas phase.
            loadings (array-like): Adsorbed phase loadings of each component (or total
                loading if total_loading=True).
            isotherms (list): List of pure component isotherm objects for each
                component.
            model (str): Name of the activity coefficient model to use. Must be one of
                the registered models in pyRAST.
            total_loading (bool, optional): If True, the input loadings are treated as
                total loadings and RAST calculations are used to fit the model
                parameters. Default is False.
            verbose (bool, optional): If True, prints convergence information during
                fitting. Default is False.
            model_parameters (dict, optional): If provided, these parameters will be
                used instead of fitting from data. Keys must match the parameter names
                for the specified model.
            c (float, optional): Sets the C parameter for fitting to single point data,
                or used as the initial guess for C when fitting to multiple data points.
                Default is 1.
            param_guess (dict, optional): Initial guess for model parameters when
                fitting from data. Keys must match the parameter names for the specified
                model. If not provided, the guess will be set to the parameter values
                that produce ideal behavior (gamma=1).
            param_bounds (dict, optional): Bounds for model parameters when fitting from
                data. Keys must match the parameter names for the specified model. If
                not provided, the bounds will be set to the default specified in the
                model class.
            optimization_options (dict, optional): Options for the least-squares
                optimization when fitting to data. This is passed directly to
                scipy.optimize.least_squares, so you can specify any options available
                there. Default is None.
            param_tol (float, optional):

        Returns:
            None: Model parameters are stored in self.model_parameters.

        Raises:
            ValueError: If input arrays have inconsistent shapes or invalid values
                or if input dictionary keys do not match model parameter names.
        """
        # Validate inputs
        partial_fug = np.asarray(partial_fug)
        loadings = np.asarray(loadings)
        if np.any(loadings < 0):
            raise ValueError('Component loadings must be non-negative.')
        if np.any(partial_fug < 0):
            raise ValueError('Partial fugacities must be non-negative.')
        if not total_loading and partial_fug.shape != loadings.shape:
            raise ValueError('Arrays for partial fugacities and loadings must have'
                            'the same dimensions.')
        if total_loading and partial_fug.shape[0] != loadings.shape[0]:
            raise ValueError('Arrays for partial fugacities and total loadings must'
                             'have the same number of data points.')
        if partial_fug.ndim == 2 and partial_fug.shape[1] != 2:
            raise ValueError('This activity coefficient model is currently only '
                            'implemented for binary mixtures (2 components). '
                            'Your arrays must have shape (n, 2) where n is the '
                            'number of data points.')
        if len(isotherms) != 2:
            raise ValueError('Exactly 2 isotherm objects must be provided in a list, '
                             'one for each component.')


        # Store data
        self.partial_fug = partial_fug
        self.loadings = loadings
        self.isotherms = isotherms
        self.c = c

        # Store model info
        self.model = model

        # Store calculation info
        self.param_tol = param_tol

        # Handle parameter guess and bounds
        self.param_bounds = dict(zip(self.param_names, self.param_default_bounds))
        if param_bounds is not None:
            if set(param_bounds.keys()) != set(self.param_names):
                raise ValueError(f'param_bounds keys must be {self.param_names}.')
            self.param_bounds = param_bounds

        self.param_guess = dict(zip(self.param_names, self.param_ideal_values +
                                                      (self.c,)))
        if param_guess is not None:
            if set(param_guess.keys()) != set(self.param_names):
                raise ValueError(f'param_guess keys must be {self.param_names}.')
            self.param_guess = param_guess
        self.param_guess = self.enforce_parameter_bounds(self.param_guess)

        # Fit model to data
        # If user provided parameters, check that keys are correct
        self.model_parameters = dict.fromkeys(self.param_names, np.nan)
        if model_parameters is not None:
            if set(model_parameters.keys()) != set(self.param_names):
                raise ValueError(f'model_parameters keys must be {self.param_names}.')
            self.model_parameters = model_parameters
        elif not total_loading:
            self._fit_component_loadings(verbose, optimization_options)
        else:
            self._fit_total_loading(verbose, optimization_options)

    def __repr__(self):
        """String representation of activity coefficient model."""
        return (f'{self.name} activity coefficient model with parameters: '
                f'{self.model_parameters}')

    def enforce_parameter_bounds(self, guess):
        """Enforces parameter bounds on the initial guess.

        Args:
            guess (dict): Initial guess for model parameters.

        Returns:
            dict: Guess with parameters enforced within bounds.
        """
        for param, value in guess.items():
            bounds = self.param_bounds[param]
            if value < bounds[0]:
                guess[param] = bounds[0]
            elif value > bounds[1]:
                guess[param] = bounds[1]
        return guess

    def ln_gamma(self, x, phi):
        """Calculates natural log of activity coefficients.

        This method is implemented in every subclass for the specific model.
        """
        raise NotImplementedError('ln_gamma method must be implemented in subclass.')

    def gamma(self, x, phi):
        """Calculates activity coefficients (gamma) from ln_gamma.

        Args:
            x (array-like): Mole fractions of components in the adsorbed phase.
            phi (float): Spreading pressure.

        Returns:
            np.ndarray: Activity coefficients for each component.
        """
        return np.exp(self.ln_gamma(x, phi))

    def inverse_excess_loading(self, x, phi):
        """Calculates inverse excess loading.

        This method is implemented in every subclass for the specific model.
        """
        raise NotImplementedError('inverse_excess_loading method must be implemented '
                                  'in subclass.')

    def _fit_total_loading(self, verbose, optimization_options):
        """Fits model parameters to total loading data.

        This method fits the model parameters to the total loading data by minimizing
        the difference between the predicted and observed total loadings. It uses a
        least-squares approach to find the optimal parameters. At each step, the
        model parameters are updated and used in a RAST calculation to predict the total
        loading.

        Note: Using total loading data can be less reliable for fitting activity
        coefficient models as multiple combinations of parameters can give similar total
        loading predictions. Only use this method if you do not have component loading
        data available.

        Args:
            verbose (bool): If True, prints model parameters at each iteration and
                            convergence information.
            optimization_options (dict): Options for the least-squares optimization when
                fitting from data. This is passed directly to
                scipy.optimize.least_squares, so you can specify any options available
                there.

        Returns:
            None: Model parameters are stored in self.model_parameters.

        Raises:
            RuntimeError: If the fit to total loading fails to converge.
        """
        # Import RAST here to avoid circular import issues
        from pyrast.calculations.rast import rast

        partial_fug = np.asarray(self.partial_fug)
        points = len(partial_fug)

        def residuals(params):
            # Assign parameters to model
            num_params = len(self.param_names)
            self.model_parameters = {self.param_names[i]: params[i]
                                     for i in range(num_params)}
            res = np.zeros(points)
            for i in range(points):
                q_total_pred = np.sum(rast(partial_fug[i], self.isotherms,
                                           self))
                res[i] = q_total_pred - self.loadings[i]
            return res

        # Assign initial guess for parameters
        initial_guess = list(self.param_guess.values())

        # Enforce parameter bounds
        bounds = [[self.param_bounds[param][0] for param in self.param_names],
                  [self.param_bounds[param][1] for param in self.param_names]]

        fitting_inputs = {
            'fun': residuals,
            'x0': initial_guess,
            'bounds': bounds,
            'xtol': self.param_tol,
        }
        if optimization_options is not None:
            fitting_inputs.update(optimization_options)
        res = least_squares(**fitting_inputs)

        if not res.success:
            raise RuntimeError(f'Total loading fit failed: {res.message}. Try a '
                               'different initial guess or check data quality.')

        # Print residuals if verbose
        if verbose:
            print(f'Fitted parameters: {dict(zip(self.param_names, res.x))}')
            print(f'Residual Sum of Squares: {res.cost}')

        # Assign final parameters to model
        num_params = len(self.param_names)
        self.model_parameters = {self.param_names[i]: res.x[i]
                                 for i in range(num_params)}

    def _fit_component_loadings(self, verbose, optimization_options):
        # Reshape data if single point is provided
        single_point = False
        if self.loadings.ndim == 1:
            single_point = True
            self.loadings = self.loadings.reshape(1, -1)
            self.partial_fug = self.partial_fug.reshape(1, -1)

        # Calculate important variables for least squares fitting
        points = len(self.partial_fug)
        q_total = np.sum(self.loadings, axis=1)
        xs = self.loadings / q_total[:, None]

        def solve_phi(x, q_total):
            p0 = np.zeros(len(self.isotherms))
            def residuals(phi, x):
                for i in range(len(self.isotherms)):
                    p0[i] = self.isotherms[i].p0(phi)

                q0 = np.array([self.isotherms[i].loading(p0[i])
                                for i in range(len(self.isotherms))])
                q_excess = self.inverse_excess_loading(x, phi)
                q_total_pred = 1.0 / (np.sum(x / q0) + q_excess)
                return q_total_pred - q_total

            # Use a bracketing strategy to ensure brentq can find a root.
            def _bracket_phi(x, residuals, phi_low = 1e-12, phi_high = 1.0,
                             max_expand = 60, phi_cap = np.inf):
                # Determine the maximum phi based on isotherm extrapolation limits
                def _phi_cap_for_isotherm(iso):
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
                caps = [_phi_cap_for_isotherm(iso) for iso in self.isotherms]
                phi_cap = min(caps) if any(np.isfinite(caps)) else np.inf

                # Check lower limit to ensure we don't start with an invalid point
                f_low = residuals(phi_low, x)
                if not np.isfinite(f_low):
                    raise ValueError('Residual is not finite at initial phi_low.')

                # Lower upper limit if necessary to find a finite residual
                f_high = residuals(phi_high, x)
                while not np.isfinite(f_high) and phi_high > phi_low:
                    phi_high *= 0.9
                    f_high = residuals(phi_high, x)

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
                    f_high = residuals(phi_high, x)

                    # if we just hit a NaN/inf, back off once
                    if not np.isfinite(f_high):
                        phi_high *= 0.5
                        f_high = residuals(phi_high, x)
                        if not np.isfinite(f_high):
                            break

                # If we exit the loop without finding a valid bracket, raise an error
                raise ValueError("Could not find valid bracketing for root finding")

            # Solve for phi with bracketed root finding
            phi_low, phi_high = _bracket_phi(x, residuals)
            return cast('float', brentq(residuals, phi_low, phi_high, args=(x,)))

        def residuals(params):
            num_params = len(params)

            self.model_parameters.update({self.param_names[i]: params[i]
                                     for i in range(num_params)})

            res = np.zeros(points * 2)

            for i in range(points):
                phi = solve_phi(xs[i], q_total[i])
                p0 = np.asarray([iso.p0(phi) for iso in self.isotherms])

                ln_gamma_exp = np.log(self.partial_fug[i] / (p0 * xs[i]))
                ln_gamma_pred = self.ln_gamma(xs[i], phi)

                res[2*i:2*i+2] = ln_gamma_pred - ln_gamma_exp
            return res

        # Assign initial guess for parameters
        initial_guess = list(self.param_guess.values())

        # Enforce parameter bounds
        bounds = [[self.param_bounds[param][0] for param in self.param_names],
                  [self.param_bounds[param][1] for param in self.param_names]]

        # Enable single point fitting if only one data point is provided
        if single_point:
            initial_guess = initial_guess[:-1]
            bounds = [bounds[0][:-1], bounds[1][:-1]]
            self.model_parameters['C'] = self.c

        fitting_inputs = {
            'fun': residuals,
            'x0': initial_guess,
            'bounds': bounds,
            'xtol': self.param_tol,
        }
        # Update fitting inputs with any user-provided optimization options
        if optimization_options is not None:
            fitting_inputs.update(optimization_options)
        res = least_squares(**fitting_inputs)

        if not res.success:
            raise RuntimeError(f'Component loading fit failed: {res.message}. Try a '
                               'different initial guess or check data quality.')
        # Print residuals if verbose
        if verbose:
            print(f'Fitted parameters: {dict(zip(self.param_names, res.x))}')
            print(f'Residual Sum of Squares: {res.cost}')

        # Assign final parameters to model
        num_params = len(res.x)
        self.model_parameters.update({self.param_names[i]: res.x[i]
                                 for i in range(num_params)})
