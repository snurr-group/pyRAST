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
    extrap_points: int

    def __init__(self, df: pd.DataFrame, loading_key: str, pressure_key: str,
                 fill_value = None, extrap_method: str | None = None,
                 extrap_p: float = 1e20, extrap_points: int = 100, **fit_options):
        """docstring"""

        # If pressure = 0 not in data frame, add it for interpolation purposes
        if 0.0 not in df[pressure_key].values:
            df = pd.concat([pd.DataFrame({pressure_key: 0.0, loading_key: 0.0},
                                         index=[0]), df])

        # Store isotherm data in self
        self.df = df.sort_values(by=pressure_key, ascending=True)
        if None in [loading_key, pressure_key]:
            raise ValueError('loading_key and pressure_key must be provided.')
        self.loading_key = loading_key
        self.pressure_key = pressure_key
        if extrap_method == 'Linear':
            extrap_method = 'linear'

        if fill_value is None:
            # If no fill value is provided, check for extrapolation method
            if extrap_method == 'linear' and extrap_method is not None:
                self.extrap_method = 'linear'
                final_load = self.df[self.loading_key].values[-1]
                final_pressure = self.df[self.pressure_key].values[-1]
                second_to_last_load = self.df[self.loading_key].values[-2]
                second_to_last_pressure = self.df[self.pressure_key].values[-2]
                slope = ((final_load - second_to_last_load) /
                         (final_pressure - second_to_last_pressure))
                next_point = final_load + slope * (extrap_p - final_pressure)
                new_row = pd.DataFrame({self.pressure_key: [extrap_p],
                                        self.loading_key: [next_point]})
                self.df = pd.concat([self.df, new_row], ignore_index=True)
                self.extrap_points = 1
            elif extrap_method in ModelIsotherm._MODELS and extrap_method is not None:
                self.extrap_method = extrap_method

                extrap_isotherm = ModelIsotherm(df=df.iloc[1:], loading_key=loading_key,
                                                pressure_key=pressure_key,
                                                model=extrap_method, **fit_options)
                iso_load = extrap_isotherm.loading(self.df[self.pressure_key].max())
                final_load = self.df[self.loading_key].values[-1]
                extrap_iso_shift = final_load - iso_load
                extrap_pressures = np.logspace(np.log10(self.df[self.pressure_key]
                                                        .max()), np.log10(extrap_p),
                                                num=extrap_points)[1:]
                extrap_loadings = (extrap_isotherm.loading(extrap_pressures)
                                    + extrap_iso_shift)
                extrap_df = pd.DataFrame({pressure_key: extrap_pressures,
                                            loading_key: extrap_loadings})
                self.df = pd.concat([self.df, extrap_df], ignore_index=True)
                self.extrap_points = extrap_points
                # except Exception as e:
                #     print(dedent(f'''
                #     The extrapolation failed when fitting the {extrap_method} model.
                #     The error message from the fitting procedure is: {e}'''))
                #     print('The extrapolation method will be set to None')
                #     self.extrap_method = None

            self.interp1d = interp1d(self.df[pressure_key], self.df[loading_key])
        else:
            self.interp1d = interp1d(self.df[pressure_key], self.df[loading_key],
                                    fill_value=fill_value, bounds_error=False)
        self.fill_value = fill_value

    def loading(self, pressure: float):
        """docstring"""

        return self.interp1d(pressure)

    def spreading_pressure(self, pressure: float):
        """docstring"""
        # throw exception if interpolating outside the range
        max_pressure = self.df[self.pressure_key].max()
        if ((self.fill_value is None) and (self.extrap_method is None)
            and (pressure > max_pressure)):
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

    def pressure(self, target_phi: float):
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
        raise ValueError( #TODO: rewrite this error message
            f'target_phi={target_phi:.4f} exceeds Φ at max pressure '
            f'({phi:.4f}). Extrapolation required.',
        )


class PCHIPInterpolatorIsotherm:
    """docstring"""
    # Instance variables
    df: pd.DataFrame
    loading_key: str
    pressure_key: str
    load_pchip: PchipInterpolator
    spread_pchip: PchipInterpolator
    p0_pchip: PchipInterpolator
    henry_const: float
    first_pressure: float
    extrap_method: str | None = None
    extrap_points: int

    def __init__(self, df: pd.DataFrame, loading_key: str, pressure_key: str,
                 grid_points: int = 500, fill_value = None,
                 extrap_method: str | None = None, extrap_p: float = 1e20,
                 extrap_points: int = 100, **fit_options):
        """docstring"""
        # Store isotherm data in self
        self.df = df.sort_values(by=pressure_key, ascending=True)
        if None in [loading_key, pressure_key]:
            raise ValueError('loading_key and pressure_key must be provided.')
        self.loading_key = loading_key
        self.pressure_key = pressure_key
        if extrap_method == 'Linear':
            extrap_method = 'linear'

        if fill_value is None:
            # If no fill value is provided, check for extrapolation method
            if extrap_method == 'linear' and extrap_method is not None:
                self.extrap_method = 'linear'
                final_load = self.df[self.loading_key].values[-1]
                final_pressure = self.df[self.pressure_key].values[-1]
                second_to_last_load = self.df[self.loading_key].values[-2]
                second_to_last_pressure = self.df[self.pressure_key].values[-2]
                slope = ((final_load - second_to_last_load) /
                         (final_pressure - second_to_last_pressure))
                next_point = final_load + slope * (extrap_p - final_pressure)
                new_row = pd.DataFrame({self.pressure_key: [extrap_p],
                                        self.loading_key: [next_point]})
                self.df = pd.concat([self.df, new_row], ignore_index=True)
                self.extrap_points = 1
            elif extrap_method in ModelIsotherm._MODELS and extrap_method is not None:
                self.extrap_method = extrap_method

                extrap_isotherm = ModelIsotherm(df=self.df, loading_key=loading_key,
                                                pressure_key=pressure_key,
                                                model=extrap_method, **fit_options)
                iso_load = extrap_isotherm.loading(self.df[self.pressure_key].max())
                final_load = self.df[self.loading_key].values[-1]
                extrap_iso_shift = final_load - iso_load
                extrap_pressures = np.logspace(np.log10(self.df[self.pressure_key]
                                                    .max()), np.log10(extrap_p),
                                                num=extrap_points)[1:]
                extrap_loadings = (extrap_isotherm.loading(extrap_pressures)
                                    + extrap_iso_shift)
                extrap_df = pd.DataFrame({pressure_key: extrap_pressures,
                                            loading_key: extrap_loadings})
                self.df = pd.concat([self.df, extrap_df], ignore_index=True)
                self.extrap_points = extrap_points - 1

        pressures = self.df[self.pressure_key].values[
                    self.df[self.pressure_key].values != 0.0]
        loadings = self.df[self.loading_key].values[
                    self.df[self.pressure_key].values != 0.0]
        self.first_pressure = pressures[0]
        self.henry_const = loadings[0] / pressures[0]
        self.load_pchip = PchipInterpolator(pressures, loadings, extrapolate=False)

        # Now calculate spreading pressure and p0 ahead of time for speed
        p_grid = np.logspace(np.log10(pressures[0]), np.log10(pressures[-1]),
                             grid_points)
        p_grid[-1] = pressures[-1]  # ensure last point is exactly the max pressure in data
        loading_grid = [self.loading(p) for p in p_grid]
        ln_p_grid = np.log(p_grid)
        spreading_grid = cumulative_trapezoid(loading_grid, ln_p_grid, initial = 0.0)
        # Build interpolators for spreading pressure and p0
        self.spread_pchip = PchipInterpolator(p_grid, spreading_grid, extrapolate=False)
        self.p0_pchip = PchipInterpolator(spreading_grid, p_grid, extrapolate=False)

    def loading(self, pressure):
        """docstring"""

        scalar_input = np.isscalar(pressure)

        pressure = np.asarray(pressure, dtype=float)

        loading = np.empty_like(pressure)

        low = pressure <= self.first_pressure
        high = ~low

        loading[low] = self.henry_const * pressure[low]
        loading[high] = self.load_pchip(pressure[high])

        if scalar_input:
            return float(loading)

        return loading

    def spreading_pressure(self, pressure: float):
        """docstring"""
        # throw exception if interpolating outside the range
        max_pressure = self.df[self.pressure_key].max()
        if pressure > max_pressure:
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

        p0 = self.first_pressure

        # Henry's law behavior enforced
        if pressure <= p0:
            return self.henry_const * pressure

        # Otherwise rely on the PCHIP interpolation of the isotherm data
        return self.henry_const * p0 + self.spread_pchip(pressure)

    def pressure(self, target_phi: float):
        """ One line description

        Args:
            param1(type): Description of param1

        Returns:
            type: Description of return value

        """
        # Enforce Henry's law behavior at low pressures
        phi_at_p0 = self.henry_const * self.first_pressure
        if target_phi <= phi_at_p0:
            return target_phi / self.henry_const

        # Otherwise rely on the PCHIP interpolation of the isotherm data
        return self.p0_pchip(target_phi)

    #     # target_phi is beyond the data range
    #     raise ValueError( #TODO: rewrite this error message
    #         f'target_phi={target_phi:.4f} exceeds Φ at max pressure '
    #         f'({phi:.4f}). Extrapolation required.',
    #     )
