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
                 c: float = 1, assume_ideal_gamma: bool = False,
                 param_guess: dict | None = None, param_bounds: dict | None = None,
                 optimization_options: dict | None = None,
                 param_tol: float = 1e-6, gamma_tol: float = 1e-4,
                 max_iter: int = 100, param_mixing: float = 0.2):
        """Initializes an activity coefficient model.

        Activity coefficient models are used in real adsorbed solution theory (RAST) to
        account for non-ideal interactions between adsorbed species. This base class
        provides the framework for fitting activity coefficient models from binary
        mixture data. There are two ways to fit the model parameters:

        1) Component loadings: If the user provides component loadings for each
        adsorbed species at different partial fugacities, the model parameters will
        be fit using an iterative approach. In the default mode, the model
        parameters are first fit assuming zero excess loading, and then iteratively
        refit with excess loading correction until convergence. Where possible, each
        model fits C and analytically solves for the other parameters to reduce
        computation time. Activity coefficient models can be fit using only a single
        data point and assuming a C value, or using 2+ data points to fit C as well.
        If the user sets assume_ideal_gamma=True, the model parameters will be fit
        assuming zero excess loading.

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

        Note: The current behavior is to print a warning if model parameters do not
        converge within max_iter iterations. Future versions may raise an exception
        instead.

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
            assume_ideal_gamma (bool, optional): If True, fits model parameters assuming
                zero excess loading. This might be faster, but is less accurate. Default
                is False.
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
            param_tol (float, optional): Tolerance for convergence of model parameters
                when fitting to data. This applies to both iterative calculations and
                least-squares optimization. Default is 1e-6.
            gamma_tol (float, optional): Tolerance for convergence of activity
                coefficients in iterative calculation to determine activity coefficients
                from component loadings. Default is 1e-4.
            max_iter (int, optional): Maximum number of iterations in iterative loops
                when fitting to data. Default is 100.
            param_mixing (float, optional): Mixing parameter for iterative fitting of
                model parameters from component loadings. A value between 0 and 1 that
                controls how much of the new parameter values are used in each
                iteration (i.e. 0.2 means 20% of new values, 80% of old). This helps
                to stabilize convergence, but will slow things down. Default is 0.2.

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
        self.gamma_tol = gamma_tol
        self.max_iter = max_iter

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
        if model_parameters is not None:
            if set(model_parameters.keys()) != set(self.param_names):
                raise ValueError(f'model_parameters keys must be {self.param_names}.')
            self.model_parameters = model_parameters
        elif not total_loading:
            self.model_parameters = dict.fromkeys(self.param_names, np.nan)
            if assume_ideal_gamma:
                excess_loading = False
                self._fit_component_loadings(excess_loading, verbose,
                                             optimization_options)
            else:
                self._fit_real_component_loadings(param_mixing, verbose,
                                                  optimization_options)
        else:
            self.model_parameters = dict.fromkeys(self.param_names, np.nan)
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

    def _gamma_from_loadings(self, comp_q, partial_fug, excess_loading):
        """Calculates gamma and phi from component loadings and partial fugacities.

        This method is used in the fitting procedure when component loadings are
        provided. It solves for the spreading pressure (phi) that corresponds to the
        provided loadings and partial fugacities, and then calculates the activity
        coefficients (gamma) from those values. If excess_loading is True, the
        spreading pressure calculation includes the excess loading correction in an
        iterative routine. It is designed to handle one point of binary data at a time.

        Args:
            comp_q (array-like): Component loadings for one data point.
            partial_fug (array-like): Partial fugacities for one data point.
            excess_loading (bool): If True, applies excess loading correction to phi
                calculation.

        Returns:
            gamma (np.ndarray): Activity coefficients for each component.
            phi (float): Spreading pressure for mixture.

        Raises:
            ValueError: If a valid bracket cannot be found for root finding phi.
        """
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
        gamma = partial_fug / (p0 * x)
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
            gamma = partial_fug / (p0 * x)

            # Check convergence
            if np.all(np.abs(gamma - gamma_old) < self.gamma_tol):
                break
        # If we reach max iterations without convergence, print warning
        else:
            print('Gamma from loadings did not converge.')

        return gamma, phi_sol

    def _fit_component_loadings(self, excess_loading, verbose, optimization_options):
        """docstring"""

        if self.loadings.ndim == 1:
            # Handle the case where a single data point is provided, thus c is assumed
            gamma, phi = self._gamma_from_loadings(self.loadings, self.partial_fug,
                                                   excess_loading)
            x = self.loadings / np.sum(self.loadings)
            c = self.c
            ln_g = np.log(gamma)

            # Set c and solve for the other model parameters with bounds
            self.model_parameters['C'] = c
            def single_residuals(p):
                for i in range(len(self.param_names)-1):
                    self.model_parameters[self.param_names[i]] = p[i]
                ln_gamma = self.ln_gamma(x, phi)
                return ln_gamma - ln_g

            # Get initial guess
            guess = list(self.param_guess.values())[:-1]

            # Get parameter bounds
            bounds = [[self.param_bounds[param][0] for param in self.param_names[:-1]],
                      [self.param_bounds[param][1] for param in self.param_names[:-1]]]

            fitting_inputs = {
                'fun': single_residuals,
                'x0': guess,
                'bounds': bounds,
                'xtol': self.param_tol,
            }
            if optimization_options is not None:
                fitting_inputs.update(optimization_options)
            sol = least_squares(**fitting_inputs)

            if not sol.success:
                raise RuntimeError(f'{self.name} model parameter fit failed with '
                                   f'message: {sol.message} Try a different initial '
                                   'guess or check data quality.')

            # Save parameters
            for i in range(len(self.param_names)-1):
                self.model_parameters[self.param_names[i]] = sol.x[i]
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
                                                             excess_loading)
                xs[i] = self.loadings[i] / np.sum(self.loadings[i])

            # Solve for all model parameters simultaneously with least squares
            def residuals(params):
                for i in range(len(self.param_names)):
                    self.model_parameters[self.param_names[i]] = params[i]

                res = np.zeros(points * 2)
                for i in range(points):
                    ln_gamma_pred = self.ln_gamma(xs[i], phi[i])
                    ln_gamma_exp = np.log(gamma[i])
                    res[2*i:2*i+2] = ln_gamma_pred - ln_gamma_exp
                return res

            # Build initial guess for least squares based on ideal case
            guess = list(self.param_guess.values())
            print(guess)

            # Enforce parameter bounds
            bounds = [[self.param_bounds[param][0] for param in self.param_names],
                      [self.param_bounds[param][1] for param in self.param_names]]

            fitting_inputs = {
                'fun': residuals,
                'x0': guess,
                'bounds': bounds,
                'xtol': self.param_tol,
            }
            if optimization_options is not None:
                fitting_inputs.update(optimization_options)
            res = least_squares(**fitting_inputs)
            if not res.success:
                raise RuntimeError(f'{self.name} model parameter fit failed with '
                                   f'message: {res.message} Try a different initial '
                                   'guess for c or check data quality.')

            # Print residuals if verbose
            if verbose:
                print(f'Fitted parameters: {dict(zip(self.param_names, res.x))}')
                print(f'Residual norm: {res.cost}')

            # Save parameters
            for i in range(len(self.param_names)):
                self.model_parameters[self.param_names[i]] = res.x[i]

    def _fit_real_component_loadings(self, param_mixing: float, verbose,
                                     optimization_options):
        """Fits model parameters to component loadings with excess loading correction.

        This method implements the outer loop of the iterative fitting procedure for
        activity coefficients when component loadings are provided. It first fits the
        model parameters using the ideal approach, and then iteratively refits the
        parameters with excess loading correction until convergence. The param_mixing
        value helps with stability by mixing the old and new parameters at each
        iteration.

        Args:
            param_mixing (float): Value between 0 and 1 to mix new and old parameters
                for stability.
            verbose (bool): If True, prints model parameters at each iteration and
                convergence information.
            optimization_options (dict): Options for the least-squares optimization when
                fitting from data. This is passed directly to
                scipy.optimize.least_squares, so you can specify any options available
                there.

        Returns:
            None: Model parameters are stored in self.model_parameters.
        """
        # First pass for model parameters is use ideal case
        excess_loading = False
        self._fit_component_loadings(excess_loading, verbose, optimization_options)

        # Now we want to iteratively get more accurate model parameters
        for iteration in range(self.max_iter):
            # Copy old parameters
            params_old = self.model_parameters.copy()
            excess_loading = True
            self._fit_component_loadings(excess_loading, verbose, optimization_options)

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
            if verbose:
                print(self.model_parameters)
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

        # Assign final parameters to model
        num_params = len(self.param_names)
        self.model_parameters = {self.param_names[i]: res.x[i]
                                 for i in range(num_params)}
