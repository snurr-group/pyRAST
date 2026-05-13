"""
functions of model isotherm parent class go here

"""
from textwrap import dedent

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class InterpolatorIsotherm:
    """docstring"""
    # Instance variables
    df: pd.DataFrame
    loading_key: str
    pressure_key: str
    interp1d: interp1d
    fill_value: float | None = None

    def __init__(self, df: pd.DataFrame, loading_key: str, pressure_key: str,
                 fill_value=None):
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

        if fill_value is None:
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
        if (self.fill_value is None) & (pressure > max_pressure):
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

#TODO: add option to extrapolate by fitting a line to the last two points
#TODO: add option to extrapolate by fitting an analytical model to the data
