"""Base class for analytical model isotherms."""
# ruff: noqa: TC002

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, least_squares, root


class ModelIsotherm:
    """Parent class for all model isotherms.

    Check the __init__ method for details on how to create an instance.
    """
    # Model list built at import time
    _MODELS = {}

    # Class variables
    name: str = ''
    param_names: tuple
    param_default_bounds: tuple

    # Instance variables
    df: pd.DataFrame
    loading_key: str
    pressure_key: str
    model_parameters: dict
    rmse: float | None = None
    model: str | None = None

    def __new__(cls, *args, **kwargs):
        """Creates an instance of the user-specified model.

        This factory design pattern allows users to follow the syntax of pyIAST while
        still providing the flexibility to use any isotherm model they choose. Users
        should never interact with this method directly.
        """
        if cls is ModelIsotherm:
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
            ModelIsotherm._MODELS[model_name] = cls

    def __init__(self, df: pd.DataFrame, loading_key: str, pressure_key: str,
                 model: str, *, model_parameters: dict | None = None,
                 param_guess: dict | None = None,
                 param_bounds: dict | None = None,
                 optimization_options: dict | None = None,
                 vst_n: int | None = None, vst_p: tuple | None = None,
                 vst_root_options: dict | None = None):
        """Initializes instances of analytical model isotherms.

        Args:
            df(pd.DataFrame): Dataframe containing isotherm data.
            loading_key(str): Column name in df corresponding to loading data.
            pressure_key(str): Column name in df corresponding to pressure data.
            model(str): Name of model to fit.
            model_parameters(dict, optional): Dictionary of model parameters. If
                provided, these parameters will be used instead of fitting to data.
                Keys must match self.param_names.
            param_guess(dict, optional): Dictionary of initial guess for fitting model
                parameters. Keys must match self.param_names. Only needed if the default
                initial guess is not sufficient for fitting.
            param_bounds(dict, optional): Dictionary of bounds for fitting model
                parameters. Keys must match self.param_names. Only needed if the default
                bounds are not sufficient for fitting.
            optimization_options(dict, optional): Dictionary of options to pass to
                scipy.optimize.least_squares. Only needed if the default optimization
                options are not sufficient for fitting.
            vst_n (int, optional): Number of points to interpolate between. A high
                number of points will use more memory but more accurately represent the
                isotherm. Default is 300 points.
            vst_p (tuple, optional): Tuple of the lowest and highest pressures to
                interpolate between. The lowest number should be a small number, but
                nonzero. Default is (1e-6, 1e40)
            vst_root_options(dict, optional): Dictionary of options to pass to
                scipy.optimize.root for solving loading from pressure. This will
                override the default options used by pyRAST. The default guess is the
                maximum loading in the input data. All options accepted by
                scipy.optimize.root are valid here.

        Raises:
            ValueError: If model is not valid, loading or pressure keys not in df,
                model_parameters keys do not match self.param_names, or param_guess keys
                do not match self.param_names.
        """
        from pyrast.isotherms import is_vst

        # Store model type
        self.model = model

        # If user provided parameters, check that keys are correct
        if model_parameters is not None:
            if set(model_parameters.keys()) != set(self.param_names):
                raise ValueError(f'model_parameters keys must be {self.param_names}.')
            self.model_parameters = model_parameters
            self.param_guess = {}
            self.rmse = None
            return

        # Check for valid inputs
        if loading_key not in df.columns:
            raise ValueError(f'Loading key {loading_key} not found.')
        if pressure_key not in df.columns:
            raise ValueError(f'Pressure key {pressure_key} not found.')

        # Store dataframe and keys
        self.df = df.sort_values(by=pressure_key, ascending=True)
        self.loading_key = loading_key
        self.pressure_key = pressure_key

        # Set initial guess
        # If user provided, check that keys are correct and enforce bounds
        self.param_guess = self.initial_guess()
        if param_guess is not None:
            if set(param_guess.keys()) != set(self.param_names):
                raise ValueError(f'param_guess keys must be {self.param_names}.')
            self.param_guess = param_guess

        # Set parameter bounds
        self.param_bounds = dict(zip(self.param_names, self.param_default_bounds))
        if param_bounds is not None:
            if set(param_bounds.keys()) != set(self.param_names):
                raise ValueError(f'param_bounds keys must be {self.param_names}.')
            self.param_bounds = param_bounds

        self.param_guess = self.enforce_parameter_bounds(self.param_guess)

        # Fit model to data
        self.model_parameters = dict.fromkeys(self.param_names, np.nan)
        if is_vst(model):
            self.vst_root_options = vst_root_options
        self._fit(optimization_options)

        # For VST only, create interpolations after fitting for speed
        if is_vst(model):
            self._initialize_vst(vst_n, vst_p)

    def __repr__(self):
        return (f'{self.name} Isotherm with parameters: {self.model_parameters}'
                f', guess: {self.param_guess}, and RMSE: {self.rmse}')

    def loading(self, pressure):
        """Returns loading as a function of pressure (or fugacity).

        This method contains a root solving scheme for VST isotherms to use during
        fitting. All other implemented isotherms have analytical expressions for
        loading.

        Args:
            pressure(float or np.ndarray): pressure(s) at which to calculate loading

        Returns:
            float or np.ndarray: loading as same variable type as input

        Raises:
            RuntimeError: If the root solving routine fails.
        """
        from pyrast.isotherms import is_vst

        # Root solving for loading during VST fitting
        if is_vst(self.model):
            # Collect all input as an array
            pressure_array = np.asarray(pressure)
            scalar_input = pressure_array.shape == ()
            pressure_values = np.atleast_1d(pressure_array).astype(float)

            # Root finding for loading for a single pressure point
            def solve_single(target_pressure):
                if target_pressure <= 0:
                    return 0.0

                def fun(x):
                    return self.pressure(x) - target_pressure
                solver_options = {
                    'fun': fun,
                    'x0': [self.df[self.loading_key].max()],
                    'method': 'lm',
                }
                if self.vst_root_options is not None:
                    solver_options.update(self.vst_root_options)

                res = root(**solver_options)
                if not res.success:
                    raise RuntimeError('Root finding failed for pressure '
                                       f'{target_pressure}. The routine failed with '
                                        f'message: {res.message}')
                return res.x.item()

            # Solve for loading at all pressure points specified
            loading = np.array([solve_single(target_pressure)
                                for target_pressure in pressure_values])
            if scalar_input:
                return loading.item()
            # Return loading array outputs in same shape as pressure input
            return loading.reshape(pressure_array.shape)

        # If any other model tries to call the parent class function, raise exception
        raise NotImplementedError('loading method not implemented for this model.')

    def spreading_pressure(self, pressure):
        """Returns spreading pressure at given pressure if implemented by subclass."""
        raise NotImplementedError('spreading_pressure method not implemented for this '
                                  'model.')

    def p0(self, target_phi):
        """Returns p0 at given spreading pressure if not implemented by subclass.

        This method works for any model without a closed form solution for p0 by using
        root finding. Root finding will be slower than a closed form solution.
        """
        p_lo = 1e-20
        p_hi = max(self.df[self.pressure_key])

        while self.spreading_pressure(p_hi) < target_phi:
            p_hi *= 10
        return brentq(lambda p: self.spreading_pressure(p) - target_phi, p_lo, p_hi)

    def pressure(self, loading):
        """Returns loading as a function of pressure (or fugacity).

        This method is only used by VST models and is overwritten by methods in the
        subclasses.
        """
        return NotImplementedError('Pressure method not implemented for this model.')

    def initial_guess(self):
        """Returns initial guess for model parameters.

        The default implementation provides a guess for the Langmuir model. Subclasses
        use this default guess as a starting point for their own initial guesses.
        """
        loading = self.df[self.loading_key].to_numpy()
        pressure = self.df[self.pressure_key].to_numpy()

        # Remove any rows with negative loading or pressure <= 0
        mask = (loading >= 0) & (pressure > 0)
        loading = loading[mask]
        pressure = pressure[mask]

        # Guess saturation loading as 10% above max loading
        m_guess = 1.1 * np.max(loading)
        # Guess K from starting point
        k_guess = loading[0] / pressure[0] / (m_guess - loading[0])
        return {'M': m_guess, 'K': k_guess}

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

    def _initialize_vst(self, vst_n, vst_p):
        """Builds loading, spreading pressure, and p0 interpolators for VST isotherms.

        This method takes some of the logic from the CubicIsotherm class to build
        interpolators without error. Interpolators are needed to make VST models run
        at a usable speed. Without interpolation, we would need to constantly apply
        root solving and numerical integration to fit activity coefficients and perform
        RAST calculations.
        """
        vst_n = vst_n if vst_n is not None else 300
        vst_p = vst_p if vst_p is not None else (1e-6, 1e40)

        # Build logarithmically spaced pressure grid to interpolate on
        p_grid = np.geomspace(vst_p[0], vst_p[1], vst_n)
        loadings = self.loading(p_grid)
        self.interp_load = PchipInterpolator(p_grid, loadings, extrapolate=False)

        # Compute spreading pressure
        ln_p_grid = np.log(p_grid)
        spreading_grid = cumulative_trapezoid(loadings, ln_p_grid, initial=0.0)

        # Guard against small negative numerical artifacts
        if np.any(spreading_grid < 0):
            spreading_grid = np.maximum(spreading_grid, 0.0)

        # Drop repeated zeros to keep inverse interpolation monotonic
        zero_indices = np.flatnonzero(spreading_grid == 0)
        if zero_indices.size > 1:
            keep_mask = np.ones_like(spreading_grid, dtype=bool)
            keep_mask[zero_indices[1:]] = False
            spreading_grid = spreading_grid[keep_mask]
            p_grid = p_grid[keep_mask]

        # Ensure strictly increasing spreading pressure for inverse interpolation
        increasing_mask = np.r_[True, np.diff(spreading_grid) > 0]
        spreading_grid = spreading_grid[increasing_mask]
        p_grid = p_grid[increasing_mask]

        # Build interpolators for spreading pressure and p0
        self.interp_spread = PchipInterpolator(p_grid, spreading_grid,
                                               extrapolate=False)
        self.interp_p0 = PchipInterpolator(spreading_grid, p_grid,
                                           extrapolate=False)

    def _fit(self, optimization_options: dict | None = None):
        """Fits the model to the data. Assigns parameters and RMSE.

        Args:
            optimization_options (dict, optional): User-specified options for the
                least squares optimization. See scipy.optimize.least_squares for
                parameter options.
        """
        # Extract loading and pressure data as numpy arrays for easier manipulation
        loading = self.df[self.loading_key].to_numpy()
        pressure = self.df[self.pressure_key].to_numpy()

        # Remove any rows with negative loading or pressure <= 0
        mask = (loading >= 0) & (pressure > 0)
        loading = loading[mask]
        pressure = pressure[mask]

        # Set up optimization problem with initial guess, bounds, and loading residual
        guess = np.array(list(self.param_guess.values()))
        bounds = [[self.param_bounds[param][0] for param in self.param_names],
                  [self.param_bounds[param][1] for param in self.param_names]]
        def residuals_loading(x):
            for i in range(len(self.param_names)):
                self.model_parameters[self.param_names[i]] = x[i]

            return loading - self.loading(pressure)

        # Build dictionary of inputs to curve fitting
        fitting_inputs = {
            'fun': residuals_loading,
            'x0': guess,
            'bounds': bounds,
        }

        # Update if user provided optimization options
        if optimization_options is not None:
            fitting_inputs.update(optimization_options)

        # Perform fitting
        result = least_squares(**fitting_inputs)

        if not result.success:
            print(result.message)
            print(f'pyRAST attempted fitting with guess: {guess} and bounds: {bounds}')
            raise RuntimeError(f'''Fitting failed for {self.model} isotherm. Try
            providing a different initial guess by passing param_guess to the
            constructor, or providing different bounds.''')

        # Store results
        self.model_parameters = dict(zip(self.param_names, result.x))
        self.rmse = np.sqrt(np.mean(result.fun**2))
