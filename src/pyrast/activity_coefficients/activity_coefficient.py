"""Base model for activity coefficient models."""

from typing import cast

import numpy as np
from scipy.optimize import least_squares, root


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
                 isotherms: list, model: str, *, global_fit: bool = True,
                 total_loading: bool = False, verbose: bool = False,
                 model_parameters: dict | None = None,
                 c: float = 1, param_guess: dict | None = None, param_tol: float = 1e-6,
                 param_bounds: dict | None = None,
                 optimization_options: dict | None = None,
                 rast_options: dict | None = None,
                 root_options: dict | None = None):
        """Initializes an activity coefficient model.

        Activity coefficient models are used in real adsorbed solution theory (RAST) to
        account for non-ideal interactions between adsorbed species. This base class
        provides the framework for fitting activity coefficient models from binary
        mixture data. There are two ways to fit the model parameters:

        1) Component loadings: If the user provides component loadings for different
        partial fugacities, the model parameters can be fit in two ways. The primary
        approach is a global fit, where the model parameters are all fit simultaneously
        by minimizing the residual between the observed and predicted component loadings
        from a RAST calculation. This approach is recommended. The secondary approach is
        a local fit, where the model parameters are fit all at once by minimizing the
        residual between the observed and predicted activity coefficients for each
        data point. Each data point requires a root finding calculation to determine
        the spreading pressure that gives the observed total loading. This approach can
        be slower, but is useful if the global approach fails to converge. Models can
        be fit with a single data point if the C parameter is fixed, or with 2+ data
        points to fit all model parameters simultaneously.

        2) Total loading: If the user provides total loadings for different partial
        fugacities, the model parameters will be fit by minimizing the residual
        between the predicted total loading from a RAST calculation and the provided
        total loading. This approach is more computationally expensive, but is
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
            param_tol (float, optional): Tolerance in parameter values for convergence
                when fitting to data. This can often be adjusted lower without
                significantly affecting the fit quality. This is passed as the 'xtol'
                parameter to scipy.optimize.least_squares. If specified again in
                optimization_options, it will override this value. Default is 1e-6.
            param_bounds (dict, optional): Bounds for model parameters when fitting from
                data. Keys must match the parameter names for the specified model. If
                not provided, the bounds will be set to the default specified in the
                model class.
            optimization_options (dict, optional): Options for the least-squares
                optimization when fitting to data. This is passed directly to
                scipy.optimize.least_squares, so you can specify any options available
                there. Default is None.
            rast_options (dict, optional): Options for the RAST solver when fitting
                to component loadings using the global approach or total loadings. This
                is passed directly to the rast method, so you can specify any options
                available there. Default is None.
            root_options (dict, optional): Options for the root-finding routine when
                fitting to component loadings using the local approach. This is passed
                directly to scipy.optimize.root, so you can specify any options
                available there. Default is None.

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
        self.param_guess = self._enforce_parameter_bounds(self.param_guess)

        # Fit model to data
        # If user provided parameters, check that keys are correct
        self.model_parameters = dict.fromkeys(self.param_names, np.nan)
        if model_parameters is not None:
            if set(model_parameters.keys()) != set(self.param_names):
                raise ValueError(f'model_parameters keys must be {self.param_names}.')
            self.model_parameters = model_parameters
        elif total_loading:
            self._fit_total_loading(verbose, optimization_options, rast_options)
        elif global_fit:
            self._fit_component_loading_global(verbose, optimization_options,
                                               rast_options)
        else:
            self._fit_component_loading_local(verbose, optimization_options,
                                              root_options)

    def __repr__(self):
        """String representation of activity coefficient model."""
        return (f'{self.name} activity coefficient model with parameters: '
                f'{self.model_parameters}')

    def _enforce_parameter_bounds(self, guess):
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

    def _fit_total_loading(self, verbose, optimization_options, rast_options):
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

        rast_inputs = {
            'warningoff': True,
            'solver_options': None,
        }
        if rast_options is not None:
            rast_inputs.update(rast_options)

        def residuals(params):
            # Assign parameters to model
            num_params = len(self.param_names)
            self.model_parameters = {self.param_names[i]: params[i]
                                     for i in range(num_params)}
            res = np.zeros(points)
            for i in range(points):
                try:
                    q_total_pred = np.sum(rast(partial_fug[i], self.isotherms, self,
                                               **rast_inputs))
                except RuntimeError as e:
                    raise RuntimeError(f'RAST calculation failed during global fitting '
                                       f'to total loadings. The error message was: '
                                       f'{e}')
                res[i] = (q_total_pred - self.loadings[i]) \
                         / np.maximum(self.loadings[i], 1e-8)
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

    def _fit_component_loading_local(self, verbose, optimization_options,
                                     root_options):
        """Fits model parameters to component loading data using a local approach.

        This method fits the model parameters to component loading data by minimizing
        the difference between the predicted and observed ln(activity coefficients).
        It uses a least-squares approach to find the optimal parameters. At each step,
        the model parameters are updated and used in a 1D root-finding calculation to
        find the spreading pressure that gives the observed total loading for each data
        point. All model parameters can be fit simultaneously using 2+ data points, or
        the C parameter can be fixed and the other parameters fit using a single data
        point.

        Args:
            verbose (bool): If True, prints model parameters and residual information
                after fitting.
            optimization_options (dict): Options for the least-squares optimization when
                fitting model parameters. This is passed directly to
                scipy.optimize.least_squares, so you can specify any options available
                there.
            root_options (dict): Options for the root-finding optimization when solving
                for phi. This is passed directly to scipy.optimize.root, so you can
                specify any options available there.
        Returns:
            None: Model parameters are stored in self.model_parameters.

        Raises:
            RuntimeError: If the fit to component loadings fails to converge or if a
                root cannot be found for the spreading pressure at any iteration.
        """
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

            # Calculate residual for determining phi by total loading at given phi
            def equations(phi, x):
                for i in range(len(self.isotherms)):
                    p0[i] = self.isotherms[i].p0(phi[0])
                q0 = np.array([self.isotherms[i].loading(p0[i])
                                for i in range(len(self.isotherms))])
                q_excess = self.inverse_excess_loading(x, phi)
                q_total_pred = 1.0 / (np.sum(x / q0) + q_excess)
                return q_total_pred - q_total

            root_inputs = {
                'fun': equations,
                'x0': 1e-6,
                'args': (x,),
                'method': 'lm',
            }
            if root_options is not None:
                root_inputs.update(root_options)

            # Solve for phi
            res = root(**root_inputs)
            if not res.success:
                raise RuntimeError(f'Root finding for phi failed: {res.message}. Try a '
                                   'different initial guess, modify the solver options,'
                                   'or check data quality.')
            return cast('float', res.x[0])

        def residuals(params):
            # Update model parameters that we are fitting
            num_params = len(params)
            self.model_parameters.update({self.param_names[i]: params[i]
                                     for i in range(num_params)})

            # Calculate residual of predicted vs observed ln(gamma)
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

    def _fit_component_loading_global(self, verbose, optimization_options,
                                      rast_options):
        """Fits model parameters to component loading data using a global approach.

        This method fits the model parameters to component loading data by minimizing
        the difference between the observed and predicted component loadings from a RAST
        calculation. It uses a least-squares approach to find the optimal parameters.
        All model parameters can be fit simultaneously using 2+ data points, or
        the C parameter can be fixed and the other parameters fit using a single data
        point.

        Args:
            verbose (bool): If True, prints model parameters and residual information
                after fitting.
            optimization_options (dict): Options for the least-squares optimization when
                fitting model parameters. This is passed directly to
                scipy.optimize.least_squares, so you can specify any options available
                there.
            rast_options (dict): Options for the RAST calculation when solving
                for phi. This is passed directly to the rast method, so you can specify
                any options available there.
        Returns:
            None: Model parameters are stored in self.model_parameters.

        Raises:
            RuntimeError: If the fit to component loadings fails to converge or if a
                rast calculation fails at any iteration.
        """
        # Import RAST here to avoid circular import issues
        from pyrast.calculations.rast import rast

        # Reshape data if single point is provided
        single_point = False
        if self.loadings.ndim == 1:
            single_point = True
            self.loadings = self.loadings.reshape(1, -1)
            self.partial_fug = self.partial_fug.reshape(1, -1)

        partial_fug = np.asarray(self.partial_fug)
        points = len(partial_fug)

        rast_inputs = {
            'warningoff': True,
            'solver_options': None,
        }
        if rast_options is not None:
            rast_inputs.update(rast_options)

        def residuals(params):
            # Assign parameters to model
            num_params = len(params)
            self.model_parameters.update({self.param_names[i]: params[i]
                                     for i in range(num_params)})
            res = np.zeros(points * 2)
            for i in range(points):
                try:
                    q_pred = rast(partial_fug[i], self.isotherms, self, **rast_inputs)
                except RuntimeError as e:
                    raise RuntimeError(f'RAST calculation failed during global fitting '
                                       f'to component loadings. The error message was: '
                                       f'{e}')
                res[2*i:2*i+2] = (q_pred - self.loadings[i]) \
                                 / np.maximum(self.loadings[i], 1e-8)
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
        if optimization_options is not None:
            fitting_inputs.update(optimization_options)
        res = least_squares(**fitting_inputs)

        if not res.success:
            raise RuntimeError(f'Component loading fit failed: {res.message} Try a '
                               'different initial guess or check data quality.')

        # Print residuals if verbose
        if verbose:
            print(f'Fitted parameters: {dict(zip(self.param_names, res.x))}')
            print(f'Residual Sum of Squares: {res.cost}')

        # Assign final parameters to model
        num_params = len(res.x)
        self.model_parameters.update({self.param_names[i]: res.x[i]
                                 for i in range(num_params)})
