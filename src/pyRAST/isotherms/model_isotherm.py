"""Base class for analytical model isotherms."""
# ruff: noqa: TC002

import numpy as np
import pandas as pd
import scipy.optimize


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

    def __new__(cls, model: str = '', *args, **kwargs):
        """Creates an instance of the user-specified model.

        This factory design pattern allows users to follow the syntax of pyIAST while
        still providing the flexibility to use any isotherm model they choose. Users
        should never interact with this method directly.
        """
        if cls is ModelIsotherm:
            if model not in cls._MODELS:
                raise ValueError(f'{model} is not a valid model. Choose from'
                                 f' {list(cls._MODELS.keys())}')
            subclass = cls._MODELS[model]
            return super().__new__(subclass)
        return super().__new__(cls)

    def __init_subclass__(cls, model_name: str = '', *args, **kwargs):
        """Registers subclasses of ModelIsotherm at import time.

        Users should never interact with this method directly. To modify the list of
        available models, edit the import statements in __init__.py.
        """
        super().__init_subclass__(**kwargs)
        if model_name:
            ModelIsotherm._MODELS[model_name] = cls

    def __init__(self, df: pd.DataFrame, loading_key: str, pressure_key: str,
                 model: str, model_parameters: dict | None = None,
                 param_guess: dict | None = None,
                 param_bounds: dict | None = None,
                 optimization_options: dict | None = None):
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

        Raises:
            ValueError: If model is not valid, loading or pressure keys not in df,
                model_parameters keys do not match self.param_names, or param_guess keys
                do not match self.param_names.

        """

        # Store model type
        self.model = model

        # If user provided parameters, check that keys are correct
        if model_parameters is not None:
            if set(model_parameters.keys()) != set(self.param_names):
                raise ValueError(f'model_parameters keys must be {self.param_names}.')
            self.model_parameters = model_parameters
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
        self._fit(optimization_options)

    def __repr__(self):
        return (f'{self.name} Isotherm with parameters: {self.model_parameters}'
                f', guess: {self.param_guess}, and RMSE: {self.rmse}')

    def loading(self, pressure):
        """Returns loading at given pressure if implemented by subclass."""
        raise NotImplementedError('loading method not implemented for this model.')

    def spreading_pressure(self, pressure):
        """Returns spreading pressure at given pressure if implemented by subclass."""
        raise NotImplementedError('spreading_pressure method not implemented.')

    def p0(self, target_phi):
        """Returns p0 at given spreading pressure if implemented by subclass."""
        raise NotImplementedError('p0 method not implemented.')

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
        def residuals(x):
            for i in range(len(self.param_names)):
                self.model_parameters[self.param_names[i]] = x[i]

            return loading - self.loading(pressure)

        # Build dictionary of inputs to curve fitting
        fitting_inputs = {
            'fun': residuals,
            'x0': guess,
            'bounds': bounds,
        }

        # Update if user provided optimization options
        if optimization_options is not None:
            fitting_inputs.update(optimization_options)

        # Perform fitting
        result = scipy.optimize.least_squares(**fitting_inputs)

        if not result.success:
            print(result.message)
            print(f'pyRAST attempted fitting with guess: {guess} and bounds: {bounds}')
            raise RuntimeError(f'''Fitting failed for {self.model} isotherm. Try
            providing a different initial guess by passing param_guess to the
            constructor, or providing different bounds.''')

        # Store results
        self.model_parameters = dict(zip(self.param_names, result.x))
        self.rmse = np.sqrt(np.mean(result.fun**2))
