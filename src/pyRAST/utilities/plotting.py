"""Utilities for plotting."""

import matplotlib.pyplot as plt
import numpy as np

from pyrast.isotherms import (
    CubicIsotherm,
    InterpolatorIsotherm,
    ModelIsotherm,
)


def plot_isotherm(isotherms, *, withfit = True, xlogscale = False, ylogscale = False,
                  pressures = None, xlim = None, ylim = None):
    """Plots isotherm data and fit for one or more isotherms.

    This function is useful for visualizing the quality of isotherm fits. It is
    important to visualize your isotherms to ensure that your data adequately covers
    the pressure range of interest and that your isotherm fits are accurate.

    Args:
        isotherms (list or isotherm object): A single isotherm or list of isotherms to
            plot.
        withfit (bool, optional): Whether to plot the isotherm fit in addition to the
            data. Default is True.
        xlogscale (bool, optional): Whether to use a logarithmic scale for the x-axis.
            Default is False.
        ylogscale (bool, optional): Whether to use a logarithmic scale for the y-axis.
            Default is False.
        pressures (array-like, optional): Array of pressures to plot the isotherm fit
            over. If not provided, the fit will be plotted over the range of pressures
            in the isotherm data. Default is None.
        xlim (tuple, optional): Tuple of the lower and upper limits for the x-axis.
            Default is None.
        ylim (tuple, optional): Tuple of the lower and upper limits for the y-axis.
            Default is None.

    Returns:
        None: This function displays a plot and does not return anything.

    Raises:
        ValueError: If isotherms is not a list or a valid isotherm object.
    """

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

        ax.scatter(df_pressures, df_loadings)
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
            ax.plot(pressure_range, loading_range, label = isotherm.name)

    ax.set_xlabel('Pressure')
    ax.set_ylabel('Loading')
    ax.legend()
    plt.show()

def plot_spreading_pressure(isotherms, *, xlogscale = False, ylogscale = False,
                            pressures = None, xlim = None, ylim = None):
    """Plots spreading pressure vs. pressure for one or more isotherms.

    This function is useful for visualizing the spreading pressure of isotherms, which
    is important for IAST and RAST calculations. Spreading pressure should be a
    continuous and monotonically increasing function of pressure.

    Args:
        isotherms (list or isotherm object): A single isotherm or list of isotherms to
            plot spreading pressure for.
        xlogscale (bool, optional): Whether to use a logarithmic scale for the x-axis.
            Default is False.
        ylogscale (bool, optional): Whether to use a logarithmic scale for the y-axis.
            Default is False.
        pressures (array-like, optional): Array of pressures to plot the spreading
            pressure integral over. If not provided, the fit will be plotted over the
            range of pressures in the isotherm data. Default is None.
        xlim (tuple, optional): Tuple of the lower and upper limits for the x-axis.
            Default is None.
        ylim (tuple, optional): Tuple of the lower and upper limits for the y-axis.
            Default is None.

    Returns:
        None: This function displays a plot and does not return anything.

    Raises:
        ValueError: If isotherms is not a list or a valid isotherm object.
    """

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
        ax.plot(pressure_range, phi_range, label = isotherm.name)

    ax.set_xlabel('Pressure')
    ax.set_ylabel('Spreading Pressure')
    ax.legend()
    plt.show()

def plot_p0(isotherms, *, xlogscale = False, ylogscale = False, pressures = None,
            xlim = None, ylim = None):
    """Plots p0 vs. spreading pressure for one or more isotherms.

    This function is useful for visualizing the p0 vs. spreading pressure relationship
    of isotherms, which is the inverse of plot_spreading_pressure.

    Args:
        isotherms (list or isotherm object): A single isotherm or list of isotherms to
            plot p0 for.
        xlogscale (bool, optional): Whether to use a logarithmic scale for the x-axis.
            Default is False.
        ylogscale (bool, optional): Whether to use a logarithmic scale for the y-axis.
            Default is False.
        pressures (array-like, optional): Array of pressures to plot p0 over. If not
            provided, the fit will be plotted over the range of pressures in the
            isotherm data. Default is None.
        xlim (tuple, optional): Tuple of the lower and upper limits for the x-axis.
            Default is None.
        ylim (tuple, optional): Tuple of the lower and upper limits for the y-axis.
            Default is None.

    Returns:
        None: This function displays a plot and does not return anything.

    Raises:
        ValueError: If isotherms is not a list or a valid isotherm object.
    """

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
        ax.plot(phi_range, p0_range, label = isotherm.name)

    ax.set_xlabel('Spreading Pressure')
    ax.set_ylabel('P0')
    ax.legend()
    plt.show()
