# ruff: noqa: BLE001
"""
functions of model isotherm parent class go here

"""

from textwrap import dedent

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import brentq

from pyrast.isotherms.model_isotherm import ModelIsotherm


class InterpolatorIsotherm:
    """docstring"""
    # Instance variables
    df: pd.DataFrame
    loading_key: str
    pressure_key: str
    interp1d: interp1d
    fill_value: float | None = None
    extrap_method: str | None = None
    extrap_p: float

    def __init__(self, df: pd.DataFrame, loading_key: str, pressure_key: str, *,
                 fill_value: float | None = None, extrap_method: str | None = None,
                 extrap_p: float = 1e20, extrap_points: int = 100, **fit_options):
        """docstring"""

        # If pressure = 0 not in data frame, add it for interpolation purposes
        if 0.0 not in df[pressure_key].values:
            df = pd.concat([pd.DataFrame({pressure_key: 0.0, loading_key: 0.0},
                                         index=[0]), df])

        # Store isotherm data in self
        df = df.sort_values(by=pressure_key, ascending=True)
        self.df = df.copy()
        if None in [loading_key, pressure_key]:
            raise ValueError('loading_key and pressure_key must be provided.')
        self.loading_key = loading_key
        self.pressure_key = pressure_key
        if extrap_method == 'Linear':
            extrap_method = 'linear'

        if fill_value is None and extrap_method is not None:
            # If no fill value is provided, check for extrapolation method
            if extrap_method == 'linear' or extrap_method in ModelIsotherm._MODELS:
                df = _build_extrapolated_df(df, loading_key, pressure_key,
                                            extrap_method, extrap_p, extrap_points,
                                            **fit_options)
                self.interp1d = interp1d(self.df[pressure_key], self.df[loading_key])
                self.extrap_p = extrap_p
            else:
                raise ValueError(f'Extrapolation method {extrap_method} not recognized.'
                                 f' Choose from "linear" or '
                                 f'{list(ModelIsotherm._MODELS.keys())}.')
        else:
            self.interp1d = interp1d(self.df[pressure_key], self.df[loading_key],
                                    fill_value=fill_value, bounds_error=False) # type:ignore
            self.fill_value = fill_value

    def loading(self, pressure: float):
        """docstring"""

        # Henry's law behavior is enforced as interpolator is linear
        return self.interp1d(pressure)

    def spreading_pressure(self, pressure: float):
        """docstring"""
        # Set max pressure to maximum pressure in df or extrapolation end
        if self.extrap_method is None:
            max_pressure = self.df[self.pressure_key].max()
        else:
            max_pressure = self.extrap_p
        # Update error message to also say increase extrap_p
        if ((self.fill_value is None) and (pressure > max_pressure)):
            raise Exception(dedent(f'''
            To compute the spreading pressure at this bulk gas pressure, we would need
            to extrapolate the isotherm since this pressure is outside the range of the
            highest pressure in your pure-component isotherm data, {max_pressure}.

            At present, your InterpolatorIsotherm object is set to throw an
            exception when this occurs, as we do not have data outside this
            pressure range to characterize the isotherm at higher pressures.

            Option 1: fit an analytical model to extrapolate the isotherm
            Option 2: pass a `fill_value` to the construction of the
                InterpolatorIsotherm object. Then, InterpolatorIsotherm will
                assume that the uptake beyond pressure {max_pressure} is equal to
                `fill_value`. This is reasonable if your isotherm data exhibits
                a plateau at the highest pressures.
            Option 3: pass an analytical model to the construction of the
                InterpolatorIsotherm object using 'extrap_method'. Then,
                InterpolatorIsotherm will use the analytical model to extrapolate the
                isotherm beyond the highest pressure in your data.
            Option 4: pass 'linear' to the construction of the InterpolatorIsotherm
                object using 'extrap_method'. Then, InterpolatorIsotherm will fit a line
                to the last two points in your data and use this line to extrapolate the
                isotherm beyond the highest pressure in your data.
            Option 3: Go back to the lab or computer to collect isotherm data
                at higher pressures. (Extrapolation can be dangerous!)
            '''))

        # Get all data points that are at nonzero pressures
        pressures = self.df[self.pressure_key].values[
                    self.df[self.pressure_key].values != 0.0]
        loadings = self.df[self.loading_key].values[
                   self.df[self.pressure_key].values != 0.0]

        # approximate loading up to first pressure point with Henry's law
        # loading = henry_const * P
        # henry_const is the initial slope in the adsorption isotherm
        henry_const = loadings[0] / pressures[0]

        # get how many of the points are less than pressure P
        n_points = np.sum(pressures < pressure)

        if n_points == 0:
            # if this pressure is between 0 and first pressure point...
            # \int_0^P henry_const P /P dP = henry_const * P ...
            return henry_const * pressure

        # P > first pressure point
        area = loadings[0]  # area of first segment \int_0^P_1 n(P)/P dP

        # get area between P_1 and P_k, where P_k < P < P_{k+1}
        for i in range(n_points - 1):
            # linear interpolation of isotherm data
            slope = (loadings[i + 1] - loadings[i]) / (pressures[i + 1] - \
                                                        pressures[i])
            intercept = loadings[i] - slope * pressures[i]
            # add area of this segment
            area += slope * (pressures[i + 1] - pressures[i]) + intercept * \
                                np.log(pressures[i + 1] / pressures[i])

        # finally, area of last segment
        slope = (self.loading(pressure) - loadings[n_points - 1]) / (
            pressure - pressures[n_points - 1])
        intercept = loadings[n_points -
                                1] - slope * pressures[n_points - 1]
        area += slope * (pressure - pressures[n_points - 1]) + intercept * \
                                np.log(pressure / pressures[n_points - 1])

        return area

    def p0(self, target_phi: float):
        """ One line description

        Args:
            param1(type): Description of param1

        Returns:
            type: Description of return value

        """
        # Get all data points that are at nonzero pressures
        pressures = self.df[self.pressure_key].values[
                    self.df[self.pressure_key].values != 0.0]
        loadings = self.df[self.loading_key].values[
                   self.df[self.pressure_key].values != 0.0]

        henry_const = loadings[0] / pressures[0]

        phi_at_p1 = henry_const * pressures[0]
        if target_phi <= phi_at_p1:
            return target_phi / henry_const

        # Accumulate phi segment by segment
        phi = phi_at_p1
        for i in range(len(pressures) - 1):
            slope = (loadings[i+1] - loadings[i]) / (pressures[i+1] - pressures[i])
            intercept = loadings[i] - slope * pressures[i]
            phi_segment = (slope * (pressures[i+1] - pressures[i])
                        + intercept * np.log(pressures[i+1] / pressures[i]))
            phi_next = phi + phi_segment

            if target_phi <= phi_next:
                # target is within this segment — brentq over [p_i, p_{i+1}] only
                phi_at_pi = phi
                def residual(p):
                    seg_area = (slope * (p - pressures[i])
                                + intercept * np.log(p / pressures[i]))
                    return phi_at_pi + seg_area - target_phi

                return brentq(residual, pressures[i], pressures[i+1])

            phi = phi_next

        # target_phi is beyond the data range
        raise RuntimeError( #TODO: rewrite this error message
            f'target_phi={target_phi:.4f} exceeds Φ at max pressure '
            f'({phi:.4f}). Extrapolation required.',
        )


class CubicIsotherm:
    """Interpolates isotherm with monotonic cubic spline."""

    # Instance variables
    df: pd.DataFrame
    loading_key: str
    pressure_key: str
    interp_load: PchipInterpolator
    interp_spread: PchipInterpolator
    interp_p0: PchipInterpolator
    henry_const: float
    first_pressure: float
    extrap_method: str | None = None
    extrap_p: float

    def __init__(self, df: pd.DataFrame, loading_key: str, pressure_key: str, *,
                 grid_points: int = 200, force_monotonic: bool = True,
                 extrap_method: str | None = None, extrap_p: float = 1e20,
                 extrap_points: int = 100, **fit_options):
        """Initializes CubicIsotherm utilizing PCHIP interpolators.

        This class uses the scipy.interpolate.PchipInterpolator, which is a monotonic
        cubic spline interpolation method. This ensures that the interpolated isotherm
        is monotonic between the points in the original data. For speed, the spreading
        pressure and p0 are calculated ahead of time on a grid and interpolated with
        Pchip as well. Extrapolation can be done with a linear fit to the last two
        points or with an analytical model fit to the data. Extrapolation can be
        dangerous but might be necessary for calculations at high bulk pressures.

        By default, the isotherm is forced to be monotonically increasing by neglecting
        any points where loading decreases with increasing pressure. This protects
        against non-physical isotherms and exceptions thrown by the Pchip interpolator.
        If your data is very noisy, this might result in a sparse isotherm. In this
        case, you can disable this feature or consider fitting an analytical model.

        Extrapolation is handled by adding extrapolate points to the original data and
        shifting the loading to ensure a continuous isotherm. The original dataframe is
        preserved for plotting while the extrapolation is saved in the interpolators.

        Args:
            df(pd.DataFrame): Dataframe containing isotherm data.
            loading_key(str): Column name in df corresponding to loading data.
            pressure_key(str): Column name in df corresponding to pressure data.
            grid_points(int, optional): Number of points to use in the spreading
                pressure and p0 interpolation grids. Default is 200, which provides a
                smooth isotherm in most cases.
            force_monotonic(bool, optional): Forces the isotherm to be monotonically
                increasing. Disable this if your data is very noisy
            extrap_method(str, optional): Method to extrapolate isotherm beyond max
                pressure. Choose from 'linear' or any implemented analytical model.
            extrap_p(float, optional): Pressure up to which to extrapolate the isotherm
                if extrap_method is not None.
            extrap_points(int, optional): Number of points to use in extrapolation if
                extrap_method is not None. Default is 100, which provides a smooth
                extrapolation in most cases.
            fit_options: Additional keyword arguments to pass to the fit of the
                analytical extrapolation model if desired. Follows syntax of
                optimization_options in ModelIsotherm.

        Raises:
            ValueError: If loading_key or pressure_key are not in df or if extrap_method
                is not recognized.
        """
        # Store isotherm data in self
        df = df.sort_values(by=pressure_key, ascending=True)

        # Preserve original df for user access and plotting
        self.df = df.copy()

        # Check for valid inputs
        if None in [loading_key, pressure_key]:
            raise ValueError('loading_key and pressure_key must be provided.')
        if loading_key not in df.columns:
            raise ValueError(f'Loading key {loading_key} not found.')
        if pressure_key not in df.columns:
            raise ValueError(f'Pressure key {pressure_key} not found.')

        self.loading_key = loading_key
        self.pressure_key = pressure_key

        # Handle extrapolation method input
        if extrap_method == 'Linear':
            extrap_method = 'linear'

        # Extrapolate data if desired
        if extrap_method is not None:
            if extrap_method == 'linear' or extrap_method in ModelIsotherm._MODELS:
                self.extrap_method = extrap_method
                self.extrap_p = extrap_p
                df = _build_extrapolated_df(df, loading_key, pressure_key,
                                            extrap_method, extrap_p, extrap_points,
                                            **fit_options)
            else:
                raise ValueError(f'Extrapolation method {extrap_method} not recognized.'
                                 f' Choose from "linear" or '
                                 f'{list(ModelIsotherm._MODELS.keys())}.')

        # Remove zero values for interpolation
        pressures = df[self.pressure_key].values[
                    df[self.pressure_key].values != 0.0]
        loadings = df[self.loading_key].values[
                    df[self.pressure_key].values != 0.0]

        # Ensure the isotherm is monotonic increasing if force_monotonic=True (default)
        if force_monotonic:
            mask = loadings >= np.maximum.accumulate(loadings)
            pressures = pressures[mask]
            loadings = loadings[mask]

        # Save information for loading interpolation
        self.first_pressure = pressures[0]
        self.henry_const = loadings[0] / pressures[0]
        self.interp_load = PchipInterpolator(pressures, loadings, extrapolate=False)

        # Now calculate spreading pressure and p0 ahead of time for speed
        p_grid = np.logspace(np.log10(pressures[0]), np.log10(pressures[-1]),
                             grid_points)
        p_grid[-1] = pressures[-1] # ensure last point is exactly  max pressure in data
        loading_grid = [self.loading(p) for p in p_grid]
        ln_p_grid = np.log(p_grid)
        spreading_grid = cumulative_trapezoid(loading_grid, ln_p_grid, initial=0.0)

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
        self.interp_p0 = PchipInterpolator(spreading_grid, p_grid, extrapolate=False)

    def loading(self, pressure):
        """Interpolates loading at given pressure with Henry's law behavior enforced.

        Handles scalar operations in calculation modules and vectorized operations for
        plotting.

        Args:
            pressure(float, np.ndarray): Pressure at which to interpolate loading.
        """

        scalar_input = np.isscalar(pressure)

        pressure = np.asarray(pressure, dtype=float)

        loading = np.empty_like(pressure)

        low = pressure <= self.first_pressure
        high = ~low

        loading[low] = self.henry_const * pressure[low]
        loading[high] = self.interp_load(pressure[high])

        if scalar_input:
            return float(loading)

        return loading

    def spreading_pressure(self, pressure: float):
        """Interpolates spreading pressure at given pressure with Henry's law enforced.

        Returns:
            float: Spreading pressure at given pressure
        Raises:
            RuntimeError if extrapolation is required.
        """

        # Max is either the max pressure in original data or extrap_p
        if self.extrap_method is None:
            max_pressure = self.df[self.pressure_key].max()
        else:
            max_pressure = self.extrap_p

        if pressure > max_pressure:
            raise RuntimeError(dedent(f'''
            To compute the spreading pressure at this bulk gas pressure, we would need
            to extrapolate the isotherm since this pressure is outside the range of the
            highest pressure in your pure-component isotherm data, {max_pressure}.

            At present, your CubicIsotherm object is set to throw an exception when this
            occurs, as we do not have data outside this pressure range to characterize
            the isotherm at higher pressures.

            If you have extrapolation enabled but are still seeing this error, increase
            the extrapolation pressure 'extrap_p' in the constructor of your
            CubicIsotherm object.

            Option 1: use an analytical model instead of interpolation.
            Option 2: pass an analytical model to the construction of the
                CubicIsotherm object using 'extrap_method'. Then, CubicIsotherm will use
                the analytical model to extrapolate the isotherm beyond the highest
                pressure in your data.
            Option 3: pass 'linear' to the construction of the CubicIsotherm object
                using 'extrap_method'. Then, CubicIsotherm will fit a line to the last
                two points in your data and use this line to extrapolate the isotherm
                beyond the highest pressure in your data.
            Option 4: Go back to the lab or computer to collect isotherm data
                at higher pressures. (Extrapolation can be dangerous!)
            '''))

        p0 = self.first_pressure

        # Henry's law behavior enforced
        if pressure <= p0:
            return self.henry_const * pressure

        # Otherwise rely on the Pchip interpolation of the isotherm data
        return self.henry_const * p0 + self.interp_spread(pressure)

    def p0(self, target_phi: float):
        """Interpolates p0 at given spreading pressure with Henry's law enforced.

        Returns:
            float: p0 at given spreading pressure
        Raises:
            RuntimeError if extrapolation is required.
        """
        # Enforce Henry's law behavior at low pressures
        phi_at_p0 = self.henry_const * self.first_pressure
        if target_phi <= phi_at_p0:
            return target_phi / self.henry_const

        # Check if target_phi is beyond the data range
        if self.extrap_method is None:
            max_pressure = self.df[self.pressure_key].max()
        else:
            max_pressure = self.extrap_p
        phi_at_max_p = self.spreading_pressure(max_pressure)
        if target_phi > phi_at_max_p:
            raise RuntimeError(dedent(f'''
            To compute p0 at this spreading pressure, we would need to extrapolate the
            isotherm since this spreading pressure is outside the range of the maximum
            spreading pressure in your pure-component isotherm data, {phi_at_max_p}.

            At present, your CubicIsotherm object is set to throw an exception when this
            occurs, as we do not have data outside this pressure range to characterize
            the isotherm at higher pressures.

            If you have extrapolation enabled but are still seeing this error, increase
            the extrapolation pressure 'extrap_p' in the constructor of your
            CubicIsotherm object.

            Option 1: use an analytical model instead of interpolation.
            Option 2: pass an analytical model to the construction of the
                CubicIsotherm object using 'extrap_method'. Then, CubicIsotherm will use
                the analytical model to extrapolate the isotherm beyond the highest
                pressure in your data.
            Option 3: pass 'linear' to the construction of the CubicIsotherm object
                using 'extrap_method'. Then, CubicIsotherm will fit a line to the last
                two points in your data and use this line to extrapolate the isotherm
                beyond the highest pressure in your data.
            Option 4: Go back to the lab or computer to collect isotherm data
                at higher pressures. (Extrapolation can be dangerous!)
            '''))

        # Otherwise rely on the Pchip interpolation of the isotherm data
        return self.interp_p0(target_phi)

def _build_extrapolated_df(df: pd.DataFrame, loading_key: str, pressure_key: str,
                          extrap_method: str, extrap_p: float, extrap_points: int,
                          **fit_options):
    """Extrapolates isotherm data in df according to extrap_method."""
    if extrap_method == 'linear':
        # Use last two points to extrapolate linearly
        final_load = df[loading_key].values[-1]
        final_pressure = df[pressure_key].values[-1]
        second_to_last_load = df[loading_key].values[-2]
        second_to_last_pressure = df[pressure_key].values[-2]
        slope = ((final_load - second_to_last_load) /
                    (final_pressure - second_to_last_pressure))
        next_point = final_load + slope * (extrap_p - final_pressure)
        new_row = pd.DataFrame({pressure_key: [extrap_p],
                                loading_key: [next_point]})
        df = pd.concat([df, new_row], ignore_index=True)

    elif extrap_method in ModelIsotherm._MODELS:
        # Fits model isotherm to data and uses it to extrapolate up to extrap_p
        extrap_isotherm = ModelIsotherm(df=df, loading_key=loading_key,
                                        pressure_key=pressure_key,
                                        model=extrap_method, **fit_options)
        iso_load = extrap_isotherm.loading(df[pressure_key].max())
        final_load = df[loading_key].values[-1]
        extrap_iso_shift = final_load - iso_load
        extrap_pressures = np.logspace(np.log10(df[pressure_key].max()),
                                       np.log10(extrap_p),
                                       num=extrap_points)[1:]
        extrap_loadings = (extrap_isotherm.loading(extrap_pressures)
                            + extrap_iso_shift)
        extrap_df = pd.DataFrame({pressure_key: extrap_pressures,
                                    loading_key: extrap_loadings})
        df = pd.concat([df, extrap_df], ignore_index=True)
    return df
