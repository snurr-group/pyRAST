"""
RAST calculation module
"""
from textwrap import dedent

import numpy as np
import scipy.optimize

from pyrast.activity_coefficients.activity_coefficient import ActivityCoefficient


def rast(partial_pressures, isotherms, activity_coefficient, *, verbose=False, warningoff=False,
         adsorbed_mole_fraction_guess = None):
    """
    docstring
    """
    partial_pressures = np.asarray(partial_pressures)
    n_components = len(isotherms)

    # Validate inputs
    if n_components <= 1:
        raise ValueError('At least two isotherms are required for RAST calculations.')
    if len(partial_pressures) != n_components:
        raise ValueError('Length of partial_pressures must match number of isotherms.')

    if verbose:
        print(f'Performing RAST calculation for {n_components} components.')
        for i in range(n_components):
            print(f'Component {i}: Partial Pressure = {partial_pressures[i]},'
                  f' Isotherm Model = {type(isotherms[i]).__name__}')

    def rast_equations(var):
        """docstring"""
        x = np.zeros(n_components)
        x[:-1] = var[:-1]
        x[-1] = 1.0 - np.sum(x[:-1])
        phi = var[-1]

        residuals = np.zeros(n_components)

        gamma = activity_coefficient.gamma(x, phi)
        for i in range(n_components):
            p0 = partial_pressures[i] / x[i] / gamma[i]
            sp = isotherms[i].spreading_pressure(p0)
            residuals[i] = phi - sp
        return residuals

    # Solve for mole fractions in adsorbed phase by equating spreading pressures
    if adsorbed_mole_fraction_guess is None:
        # Default guess: pure-component loadings at these partial pressures
        loading_guess = [isotherms[i].loading(partial_pressures[i]) for i in \
                                                                    range(n_components)]
        loading_guess = np.asarray(loading_guess)
        adsorbed_mole_fraction_guess = loading_guess / np.sum(loading_guess)
    else:
        np.testing.assert_almost_equal(1.0, np.sum(adsorbed_mole_fraction_guess),
                                       decimal=4)
        # if list convert to numpy array
        adsorbed_mole_fraction_guess = np.asarray(adsorbed_mole_fraction_guess)

    guess = np.concatenate((adsorbed_mole_fraction_guess[:-1], [0.0]))  # initial guess for phi is 0

    res = scipy.optimize.root(rast_equations,
                              guess,
                              method='lm')

    if not res.success:
        print(res.message)
        raise RuntimeError(dedent('''\
                        Root finding for adsorbed phase mole fractions
                        failed. This is likely because the default guess is not good
                        enough. Try a different starting guess for the adsorbed phase
                        mole fractions by passing an array adsorbed_mole_fraction_guess
                        '''))

    adsorbed_mole_fractions = np.asarray(res.x[:-1])
    phi = res.x[-1]

    # Concatenate mole fraction of last component
    adsorbed_mole_fractions = np.concatenate((adsorbed_mole_fractions,
                                             [1.0 - np.sum(adsorbed_mole_fractions)]))

    if np.any((adsorbed_mole_fractions < 0.0) | (adsorbed_mole_fractions > 1.0)):
        raise ValueError(dedent('''\
                         Adsorbed mole fraction not in [0, 1]. Try a different
                         starting guess for the adsorbed mole fractions by passing an
                         array or list 'adsorbed_mole_fraction_guess' to this function.
                         e.g. adsorbed_mole_fraction_guess=[0.2, 0.8]'''))

    pressure0 = partial_pressures / adsorbed_mole_fractions / activity_coefficient.gamma(adsorbed_mole_fractions, phi)

    # Solve for total gas adsorbed
    inverse_loading = 0.0
    for i in range(n_components):
        inverse_loading += (adsorbed_mole_fractions[i] /
                            isotherms[i].loading(pressure0[i]))
    inverse_loading += activity_coefficient.inverse_excess_loading(adsorbed_mole_fractions, phi)
    loading_total = 1.0 / inverse_loading

    # get loading of each component by multiplying by mole fractions
    loadings = adsorbed_mole_fractions * loading_total
    if verbose:
        # print RAST loadings and corresponding pure-component loadings
        for i in range(n_components):
            print('Component ', i)
            print('\tp = ', partial_pressures[i])
            print('\tp^0 = ', pressure0[i])
            print('\tLoading: ', loadings[i])
            print('\tx = ', adsorbed_mole_fractions[i])
            print('\tSpreading pressure = ',
                  isotherms[i].spreading_pressure(pressure0[i]))

    # print warning if had to extrapolate isotherm in spreading pressure
    if not warningoff:
        for i in range(n_components):
            max_pressure = isotherms[i].df[isotherms[i].pressure_key].max()
            if pressure0[i] > max_pressure:
                print(dedent(f'''\
                WARNING:
                Component {i}: p^0 = {pressure0[i]} > {max_pressure}, the highest
                pressure exhibited in the pure-component isotherm data. Thus, pyRAST had
                to extrapolate the isotherm data to achieve this RAST result.'''))

    # return loadings [component 1, component 2, ...]. same units as in data
    return loadings
