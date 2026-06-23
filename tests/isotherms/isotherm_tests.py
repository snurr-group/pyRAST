import numpy as np
import pandas as pd

from pyrast.calculations.iast import iast
from pyrast.isotherms import CubicIsotherm, InterpolatorIsotherm, ModelIsotherm
from pyrast.utilities.plotting import plot_isotherm, plot_p0, plot_spreading_pressure

ch4_data = pd.read_csv('/Users/jonah/Desktop/Research/IAST Project/pyRAST/tests/test_inputs/Cu-BTC_data/2014[Cu][tbo]3[ASR]7/2014[Cu][tbo]3[ASR]7_CH4_298K_isotherm.csv')
#ch4_data = ch4_data[ch4_data['total_pressure[Pa]'] < 1e7]
ch4_fit_isotherm = ModelIsotherm(df=ch4_data, loading_key='CH4_uptake_absolute[mol/kg]',
                             pressure_key='CH4_fugacity[Pa]', model='W-VST')
plot_isotherm(ch4_fit_isotherm, xlogscale=True)
print(ch4_fit_isotherm)
#plot_isotherm(ch4_fit_isotherm, xlogscale=True)

ch4_isotherm = ModelIsotherm(df=ch4_data, loading_key='CH4_uptake_absolute[mol/kg]',
                             pressure_key='total_pressure[Pa]', model='Langmuir',
                             model_parameters={'M': 1.0, 'K': 0.5})
print(ch4_isotherm)

# ch4_isotherm = ModelIsotherm(df=ch4_data, loading_key='CH4_uptake_absolute[mol/kg]',
#                              pressure_key='total_pressure[Pa]', model='Langmuir',
#                              model_parameters={'K': 1.0, 'K': 0.5})
# print(ch4_isotherm)

ch4_isotherm = ModelIsotherm(df=ch4_data, loading_key='CH4_uptake_absolute[mol/kg]',
                             pressure_key='total_pressure[Pa]', model='Langmuir',
                             model_parameters={'M': 1.0, 'K': 0.5},
                             param_guess={'M': 0.5, 'K': 0.25})
print(ch4_isotherm)

ch4_isotherm1 = ModelIsotherm(df=ch4_data, loading_key='CH4_uptake_absolute[mol/kg]',
                             pressure_key='total_pressure[Pa]', model='Langmuir',
                             model_parameters={'M': 1.0, 'K': 0.5},
                             param_guess={'M': 20, 'K': 0.25},
                             param_bounds={'M': (0.1, 10.0), 'K': (0.01, 1.0)})
print(ch4_isotherm)

ch4_isotherm = InterpolatorIsotherm(df=ch4_data, loading_key='CH4_uptake_absolute[mol/kg]',
                                   pressure_key='total_pressure[Pa]', extrap_method='Langmuir')
print(ch4_isotherm)

#print(iast([1e5, 1e5], [ch4_isotherm1, ch4_isotherm]))

ch4_pchip_isotherm = CubicIsotherm(df=ch4_data, loading_key='CH4_uptake_absolute[mol/kg]',
                                              pressure_key='total_pressure[Pa]', extrap_method='Langmuir')

pressures = np.logspace(3, 13, 100)
plot_isotherm(ch4_pchip_isotherm, xlogscale=True, pressures=pressures)
# pressures = np.logspace(3, 7, 100)
# plot_p0([ch4_isotherm, ch4_pchip_isotherm], xlogscale = True, pressures=pressures)
