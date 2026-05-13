"""
IAST calculation module
"""
# ruff: noqa: F401
from textwrap import dedent

import numpy as np
import scipy.optimize

from pyrast.isotherms.model_isotherm import ModelIsotherm


def iast(partial_pressures, isotherms, *, verbose=False, warningoff=False,
         adsorbed_mole_fraction_guess = None):
    """
    docstring
    """
    partial_pressures = np.asarray(partial_pressures)
    n_components = len(isotherms)

    # Validate inputs
    if n_components <= 1:
        raise ValueError("At least two isotherms are required for IAST calculations.")
    if len(partial_pressures) != n_components:
        raise ValueError("Length of partial_pressures must match number of isotherms.")

    if verbose:
        print(f"Performing IAST calculation for {n_components} components.")
        for i in range(n_components):
            print(f'Component {i}: Partial Pressure = {partial_pressures[i]},'
                  f' Isotherm Model = {type(isotherms[i]).__name__}')

    # Assert that the spreading pressures of each component are equal
    def spreading_pressure_differences(adsorbed_mole_fractions):
        """docstring"""
        diff = np.zeros((n_components - 1,))
        for i in range(n_components - 1):
            if i == n_components - 2:
                # automatically assert \sum z_i = 1
                ads_mol_frac2 = 1.0 - np.sum(adsorbed_mole_fractions)
            else:
                ads_mol_frac2 = adsorbed_mole_fractions[i + 1]
            sp1 = isotherms[i].spreading_pressure(partial_pressures[i] /
                                                  adsorbed_mole_fractions[i])
            sp2 = isotherms[i + 1].spreading_pressure(partial_pressures[i + 1] /
                                                      ads_mol_frac2)
            diff[i] = sp1 - sp2
        return diff

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

    res = scipy.optimize.root(spreading_pressure_differences,
                              adsorbed_mole_fraction_guess[:-1],
                              method='lm')

    if not res.success:
        print(res.message)
        raise RuntimeError(dedent('''\
                        Root finding for adsorbed phase mole fractions
                        failed. This is likely because the default guess is not good
                        enough. Try a different starting guess for the adsorbed phase
                        mole fractions by passing an array adsorbed_mole_fraction_guess
                        '''))

    adsorbed_mole_fractions = res.x

    # Concatenate mole fraction of last component
    adsorbed_mole_fractions = np.concatenate((adsorbed_mole_fractions,
                                             [1.0 - np.sum(adsorbed_mole_fractions)]))

    if np.any((adsorbed_mole_fractions < 0.0) | (adsorbed_mole_fractions > 1.0)):
        raise ValueError(dedent('''\
                         Adsorbed mole fraction not in [0, 1]. Try a different
                         starting guess for the adsorbed mole fractions by passing an
                         array or list 'adsorbed_mole_fraction_guess' to this function.
                         e.g. adsorbed_mole_fraction_guess=[0.2, 0.8]'''))

    pressure0 = partial_pressures / adsorbed_mole_fractions

    # Solve for total gas adsorbed
    inverse_loading = 0.0
    for i in range(n_components):
        inverse_loading += (adsorbed_mole_fractions[i] /
                            isotherms[i].loading(pressure0[i]))
    loading_total = 1.0 / inverse_loading

    # get loading of each component by multiplying by mole fractions
    loadings = adsorbed_mole_fractions * loading_total
    if verbose:
        # print IAST loadings and corresponding pure-component loadings
        for i in range(n_components):
            print("Component ", i)
            print("\tp = ", partial_pressures[i])
            print("\tp^0 = ", pressure0[i])
            print("\tLoading: ", loadings[i])
            print("\tx = ", adsorbed_mole_fractions[i])
            print("\tSpreading pressure = ",
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
                to extrapolate the isotherm data to achieve this IAST result.'''))

    # return loadings [component 1, component 2, ...]. same units as in data
    return loadings

def reverse_iast(adsorbed_mole_fractions, total_pressure, isotherms, *, verbose=False,
                 warningoff=False, gas_mole_fraction_guess=None):
    """docstring"""
    n_components = len(isotherms)
    adsorbed_mole_fractions = np.asarray(adsorbed_mole_fractions)
    if n_components <= 1:
        raise ValueError("At least two isotherms are required for IAST calculations.")

    if np.size(adsorbed_mole_fractions) != n_components:
        print('''Example use:\n
              reverse_IAST([0.5,0.5], 1.0, [xe_isotherm, kr_isotherm],
              verboseflag=true)''')
        raise ValueError("Length of adsorbed_mole_fractions != number of isotherms.")

    if np.sum(adsorbed_mole_fractions) != 1.0:
        raise ValueError("Sum of adsorbed mole fractions must be 1.0")

    if verbose:
        print(f"Performing reverse IAST calculation for {n_components} components.")
        for i in range(n_components):
            print(f'Desired adsorbed mole fraction of component {i} = '
                  f'{adsorbed_mole_fractions[i]}')

    # assert that the spreading pressures of each component are equal
    def spreading_pressure_differences(gas_mole_fractions):
        """docstring"""
        diff = np.zeros((n_components - 1, ))
        for i in range(n_components - 1):
            if i == n_components - 2:
                # automatically assert \sum y_i = 1
                gas_mole_fraction_n = 1.0 - np.sum(gas_mole_fractions)
            else:
                gas_mole_fraction_n = gas_mole_fractions[i + 1]

            sp1 = isotherms[i].spreading_pressure(
                total_pressure * gas_mole_fractions[i] / adsorbed_mole_fractions[i])

            sp2 = isotherms[i + 1].spreading_pressure(
                total_pressure * gas_mole_fraction_n / adsorbed_mole_fractions[i + 1])
            diff[i] = sp1 - sp2

        return diff

    # Solve for mole fractions in gas phase by equating spreading pressures
    if gas_mole_fraction_guess is None:
        # Default guess: adsorbed mole fraction
        gas_mole_fraction_guess = adsorbed_mole_fractions
    else:
        np.testing.assert_almost_equal(1.0, np.sum(gas_mole_fraction_guess), decimal=4)
        gas_mole_fraction_guess = np.asarray(gas_mole_fraction_guess)

    res = scipy.optimize.root(spreading_pressure_differences,
                              gas_mole_fraction_guess[:-1],
                              method='lm')

    if not res.success:
        print(res.message)
        raise RuntimeError(dedent('''\
                Root finding for gas phase mole fractions failed. This is likely
                because the default guess in pyRAST is not good enough. Try a
                different starting guess for the gas phase mole fractions by
                passing an array or list gas_mole_fraction_guess to this function.'''))

    gas_mole_fractions = res.x

    # concatenate mole fraction of last component
    gas_mole_fractions = np.concatenate((gas_mole_fractions,
                                         np.array([1.0 - np.sum(gas_mole_fractions)])))

    if np.any((gas_mole_fractions < 0.0) | (gas_mole_fractions > 1.0)):
        raise ValueError(dedent('''\
                         Gas mole fraction not in [0, 1]. Try a different
                         starting guess for the gas mole fractions by passing an
                         array or list 'gas_mole_fraction_guess' to this function.
                         e.g. gas_mole_fraction_guess=[0.2, 0.8]'''))

    pressure0 = total_pressure * gas_mole_fractions / adsorbed_mole_fractions

    # solve for the total gas adsorbed
    inverse_loading= 0.0
    for i in range(n_components):
        inverse_loading += (adsorbed_mole_fractions[i] /
                            isotherms[i].loading(pressure0[i]))
    loading_total = 1.0 / inverse_loading

    # get loading of each component by multiplying by mole fractions
    loadings = adsorbed_mole_fractions * loading_total

    if verbose:
        # print IAST loadings and corresponding pure-component loadings
        for i in range(n_components):
            print("Component ", i)
            print("\tDesired mole fraction in adsorbed phase, x = ", \
                adsorbed_mole_fractions[i])
            print("\tBulk gas mole fraction that gives this, y = ", \
                gas_mole_fractions[i])
            print("\tSpreading pressure = ", \
                isotherms[i].spreading_pressure(pressure0[i]))
            print("\tp^0 = ", pressure0[i])
            print("\tLoading: ", loadings[i])

    # print warning if had to extrapolate isotherm in spreading pressure
    if not warningoff:
        for i in range(n_components):
            max_pressure = isotherms[i].df[isotherms[i].pressure_key].max()
            if pressure0[i] > max_pressure:
                print(dedent(f'''\
                WARNING:
                Component {i}: p^0 = {pressure0[i]} > {max_pressure}, the highest
                pressure exhibited in the pure-component isotherm data. Thus, pyRAST had
                to extrapolate the isotherm data to achieve this IAST result.'''))

    # return mole fractions in gas phase, component loadings
    return gas_mole_fractions, loadings
