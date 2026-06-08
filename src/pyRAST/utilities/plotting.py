"""
Utilities for plotting.
"""

import matplotlib.pyplot as plt
import numpy as np

from pyrast.isotherms import (
    CubicIsotherm,
    InterpolatorIsotherm,
    ModelIsotherm,
)


def plot_isotherm(isotherms, *, withfit = True, xlogscale = False,
                  ylogscale = False, pressures = None, xlim = None, ylim = None):
    """docstring"""

    if isinstance(isotherms,
                  (ModelIsotherm, InterpolatorIsotherm, CubicIsotherm)):
        isotherms = [isotherms]

    fig, ax = plt.subplots(layout = 'constrained', figsize = (6, 4))

    if xlogscale:
        ax.set_xscale('log')
    if ylogscale:
        ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    for num, isotherm in enumerate(isotherms):
        df_pressures = isotherm.df[isotherm.pressure_key].values[
                    isotherm.df[isotherm.pressure_key].values != 0.0]
        df_loadings = isotherm.df[isotherm.loading_key].values[
                   isotherm.df[isotherm.pressure_key].values != 0.0]
        if pressures is not None:
            mask = (df_pressures >= pressures.min()) & (df_pressures <= pressures.max())
            df_pressures = df_pressures[mask]
            df_loadings = df_loadings[mask]

        ax.scatter(df_pressures, df_loadings, label = f'Isotherm {num + 1} data')
        if withfit:
            if pressures is not None:
                pressure_range = pressures
            else:
                pressure_range = np.logspace(np.log10(df_pressures.min()),
                                            np.log10(df_pressures.max()),
                                            100)
            loading_range = np.zeros(len(pressure_range))
            for i, p in enumerate(pressure_range):
                loading_range[i] = isotherm.loading(p)
            ax.plot(pressure_range, loading_range, label = f'Isotherm {num + 1} fit')

    ax.set_xlabel('Pressure')
    ax.set_ylabel('Loading')
    ax.legend()
    plt.show()

def plot_spreading_pressure(isotherms, *, xlogscale = False,
                  ylogscale = False, pressures = None, xlim = None, ylim = None):
    """docstring"""

    if isinstance(isotherms,
                  (ModelIsotherm, InterpolatorIsotherm, CubicIsotherm)):
        isotherms = [isotherms]

    fig, ax = plt.subplots(layout = 'constrained', figsize = (6, 4))

    if xlogscale:
        ax.set_xscale('log')
    if ylogscale:
        ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    for num, isotherm in enumerate(isotherms):
        df_pressures = isotherm.df[isotherm.pressure_key].values[
                    isotherm.df[isotherm.pressure_key].values != 0.0]
        if pressures is not None:
            mask = (df_pressures >= pressures.min()) & (df_pressures <= pressures.max())
            df_pressures = df_pressures[mask]
            pressure_range = pressures
        else:
            pressure_range = np.logspace(np.log10(df_pressures.min()),
                                        np.log10(df_pressures.max()),
                                        100)
        phi_range = np.zeros(len(pressure_range))
        for i, p in enumerate(pressure_range):
            phi_range[i] = isotherm.spreading_pressure(p)
        ax.plot(pressure_range, phi_range, label = f'Isotherm {num + 1} fit')

    ax.set_xlabel('Pressure')
    ax.set_ylabel('Spreading Pressure')
    ax.legend()
    plt.show()

def plot_p0(isotherms, *, xlogscale = False,
                  ylogscale = False, pressures = None, xlim = None, ylim = None):
    """docstring"""

    if isinstance(isotherms,
                  (ModelIsotherm, InterpolatorIsotherm, CubicIsotherm)):
        isotherms = [isotherms]

    fig, ax = plt.subplots(layout = 'constrained', figsize = (6, 4))

    if xlogscale:
        ax.set_xscale('log')
    if ylogscale:
        ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    for num, isotherm in enumerate(isotherms):
        df_pressures = isotherm.df[isotherm.pressure_key].values[
                    isotherm.df[isotherm.pressure_key].values != 0.0]

        if pressures is not None:
            mask = (df_pressures >= pressures.min()) & (df_pressures <= pressures.max())
            df_pressures = df_pressures[mask]
        if pressures is not None:
            pressure_range = pressures
        else:
            pressure_range = np.logspace(np.log10(df_pressures.min()),
                                        np.log10(df_pressures.max()),
                                        100)
        phi_range = np.zeros(len(pressure_range))
        for i, p in enumerate(pressure_range):
            phi_range[i] = isotherm.spreading_pressure(p)
        p0_range = np.zeros(len(pressure_range))
        for i, phi in enumerate(phi_range):
            p0_range[i] = isotherm.pressure(phi) #type: ignore
        ax.plot(phi_range, p0_range, label = f'Isotherm {num + 1} fit')

    ax.set_xlabel('Spreading Pressure')
    ax.set_ylabel('P0')
    ax.legend()
    plt.show()
