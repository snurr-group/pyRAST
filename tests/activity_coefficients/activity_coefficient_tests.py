import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pyrast.isotherms import CubicIsotherm, ModelIsotherm, InterpolatorIsotherm
from pyrast.activity_coefficients import ActivityCoefficient
from pyrast.utilities.plotting import plot_isotherm, plot_spreading_pressure, plot_p0
from pyrast.calculations.rast import rast
from pyrast.calculations.iast import iast

comp1_data = pd.read_csv('/Users/jonah/Desktop/Research/IAST Project/pyRAST/tests/test_inputs/Results/0000[Cd][nan]3[ASR]12/0000[Cd][nan]3[ASR]12_CH4_298K_isotherm.csv')
comp2_data = pd.read_csv('/Users/jonah/Desktop/Research/IAST Project/pyRAST/tests/test_inputs/Results/0000[Cd][nan]3[ASR]12/0000[Cd][nan]3[ASR]12_CO2_298K_isotherm.csv')

binary_data1 = pd.read_csv('/Users/jonah/Desktop/Research/IAST Project/pyRAST/tests/test_inputs/Results/0000[Cd][nan]3[ASR]12/0000[Cd][nan]3[ASR]12_CH4_CO2_10.0_90.0_298K_isotherm.csv')
binary_data2 = pd.read_csv('/Users/jonah/Desktop/Research/IAST Project/pyRAST/tests/test_inputs/Results/0000[Cd][nan]3[ASR]12/0000[Cd][nan]3[ASR]12_CH4_CO2_25.0_75.0_298K_isotherm.csv')
binary_data3 = pd.read_csv('/Users/jonah/Desktop/Research/IAST Project/pyRAST/tests/test_inputs/Results/0000[Cd][nan]3[ASR]12/0000[Cd][nan]3[ASR]12_CH4_CO2_50.0_50.0_298K_isotherm.csv')
binary_data4 = pd.read_csv('/Users/jonah/Desktop/Research/IAST Project/pyRAST/tests/test_inputs/Results/0000[Cd][nan]3[ASR]12/0000[Cd][nan]3[ASR]12_CH4_CO2_75.0_25.0_298K_isotherm.csv')
binary_data5 = pd.read_csv('/Users/jonah/Desktop/Research/IAST Project/pyRAST/tests/test_inputs/Results/0000[Cd][nan]3[ASR]12/0000[Cd][nan]3[ASR]12_CH4_CO2_90.0_10.0_298K_isotherm.csv')
total_f1 = (binary_data1['CH4_fugacity[Pa]'] + binary_data1['CO2_fugacity[Pa]']).to_numpy()
total_f2 = (binary_data2['CH4_fugacity[Pa]'] + binary_data2['CO2_fugacity[Pa]']).to_numpy()
total_f3 = (binary_data3['CH4_fugacity[Pa]'] + binary_data3['CO2_fugacity[Pa]']).to_numpy()
total_f4 = (binary_data4['CH4_fugacity[Pa]'] + binary_data4['CO2_fugacity[Pa]']).to_numpy()
total_f5 = (binary_data5['CH4_fugacity[Pa]'] + binary_data5['CO2_fugacity[Pa]']).to_numpy()
total_f = np.concatenate((total_f1, total_f2, total_f3, total_f4, total_f5))
y1 = [[0.1, 0.9]] * len(binary_data1)
y2 = [[0.25, 0.75]] * len(binary_data2)
y3 = [[0.5, 0.5]] * len(binary_data3)
y4 = [[0.75, 0.25]] * len(binary_data4)
y5 = [[0.9, 0.1]] * len(binary_data5)
y = np.concatenate((y1, y2, y3, y4, y5))
comp_q1 = binary_data1[['CH4_uptake_absolute[mol/kg]', 'CO2_uptake_absolute[mol/kg]']].to_numpy()
comp_q2 = binary_data2[['CH4_uptake_absolute[mol/kg]', 'CO2_uptake_absolute[mol/kg]']].to_numpy()
comp_q3 = binary_data3[['CH4_uptake_absolute[mol/kg]', 'CO2_uptake_absolute[mol/kg]']].to_numpy()
comp_q4 = binary_data4[['CH4_uptake_absolute[mol/kg]', 'CO2_uptake_absolute[mol/kg]']].to_numpy()
comp_q5 = binary_data5[['CH4_uptake_absolute[mol/kg]', 'CO2_uptake_absolute[mol/kg]']].to_numpy()
comp_q = np.concatenate((comp_q1, comp_q2, comp_q3, comp_q4, comp_q5))
partial_fug = total_f[:, np.newaxis] * y
# comp_q = np.sum(comp_q, axis=1)

# comp1_isotherm = ModelIsotherm(comp1_data.iloc[:], 'CH4_uptake_absolute[mol/kg]',
#                              'CH4_fugacity[Pa]', 'BET')
# print(comp1_isotherm)

# comp2_isotherm = ModelIsotherm(df=comp2_data.iloc[:], loading_key='CO2_uptake_absolute[mol/kg]',
#                              pressure_key='CO2_fugacity[Pa]', model='W-VST')
# plot_isotherm([comp1_isotherm, comp2_isotherm], xlogscale=True)
comp1_isotherm = CubicIsotherm(df=comp1_data[:], loading_key='CH4_uptake_absolute[mol/kg]',
                             pressure_key='CH4_fugacity[Pa]', extrap_method='Langmuir', extrap_p = 1e40)
comp2_isotherm = CubicIsotherm(df=comp2_data[:], loading_key='CO2_uptake_absolute[mol/kg]',
                             pressure_key='CO2_fugacity[Pa]', extrap_method='Langmuir', extrap_p = 1e40)

# total_f = np.asarray([2499+7496, 49896.22+49733.74, 489865.6+473764.2])
# y = np.asarray([[0.25, 0.75], [0.5, 0.5], [0.5, 0.5]])
# comp_q = np.asarray([[0.0393, 0.622], [0.447, 2.875], [1.13, 8.16]])
# partial_fug = total_f[:, np.newaxis] * y
# total_f = np.asarray([49896.22+49733.74])
# y = [0.5, 0.5]
# comp_q = [0.447, 2.875]
# partial_fug = total_f * np.array(y)
isotherms = [comp1_isotherm, comp2_isotherm]
model = 'Wilson'
start_time = time.perf_counter()
ac = ActivityCoefficient(partial_fug=partial_fug, loadings=comp_q, isotherms=isotherms,
                         model=model, verbose=True, total_loading=False, param_tol=1e-6)
print(ac)
end_time = time.perf_counter()
print(f'Fitting took {end_time - start_time:.4f} seconds')
# x_range = np.linspace(0.001, 0.999, 100)
# gamma_values = np.array([ac.gamma([x_i, 1-x_i], total_f) for x_i in x_range])
# plt.plot(x_range, gamma_values[:, 0], label='gamma_0')
# plt.plot(x_range, gamma_values[:, 1], label='gamma_1')
# plt.xlabel('x_0')
# plt.ylabel('Activity Coefficient')
# plt.title('Activity Coefficients vs. Mole Fraction')
# plt.legend()

#print(rast(partial_pressures=[total_f * y_i for y_i in y], isotherms=isotherms, activity_coefficient=ac, warningoff=True))
#print(iast(partial_pressures=[total_f * y_i for y_i in y], isotherms=isotherms, warningoff=True))

i = 0
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()
y_range = np.linspace(0.001, 0.999, 100)
iast_start = time.perf_counter()
for y_i in y_range:
    comp_q = iast(partial_pressures=[total_f[i] * y_i, total_f[i] * (1-y_i)], isotherms=isotherms, warningoff=True)
    x_0 = comp_q[0] / np.sum(comp_q)
    ax.plot(y_i, x_0, 'bo')
    ax1.plot(y_i, comp_q[1], 'bo')
ax.set_xlabel('Gas Phase Mole Fraction of Component 0')
ax.set_ylabel('Adsorbed Phase Mole Fraction of Component 0')
ax1.set_xlabel('Gas Phase Mole Fraction of Component 0')
ax1.set_ylabel('Adsorbed Phase Loading of Component 0')
print(f'IAST calculations took {time.perf_counter() - iast_start:.4f} seconds')

rast_start = time.perf_counter()
for y_i in y_range:
    try:
        comp_q = rast(partial_pressures=[total_f[i] * y_i, total_f[i] * (1-y_i)], isotherms=isotherms, activity_coefficient=ac, warningoff=True)
        x_0 = comp_q[0] / np.sum(comp_q)
        ax.plot(y_i, x_0, 'ro')
        ax1.plot(y_i, comp_q[1], 'ro')
    except Exception as e:
        print(e)
print(f'RAST calculations took {time.perf_counter() - rast_start:.4f} seconds')
ax.legend(['IAST', 'RAST'])

# total_f = 489865.6+473764.2
# y = [0.5, 0.5]
# comp_q = [1.13, 8.16]
# isotherms = [comp1_isotherm, comp2_isotherm]
# model = 'aAC'

# ac1 = ActivityCoefficient(total_f=total_f, y=y, comp_q=comp_q, isotherms=isotherms, model=model)
# ac1._fit_to_gamma()
# #print(rast(partial_pressures=[total_f * y_i for y_i in y], isotherms=isotherms, activity_coefficient=ac1, warningoff=True))
# #print(iast(partial_pressures=[total_f * y_i for y_i in y], isotherms=isotherms, warningoff=True))


# # x_range = np.linspace(0.001, 0.999, 100)
# # gamma_values = np.array([ac1.gamma([x_i, 1-x_i], total_f) for x_i in x_range])
# # plt.plot(x_range, gamma_values[:, 0], label='gamma_0')
# # plt.plot(x_range, gamma_values[:, 1], label='gamma_1')
# # plt.xlabel('x_0')
# # plt.ylabel('Activity Coefficient')
# # plt.title('Activity Coefficients vs. Mole Fraction')
# # plt.legend()
# # plt.ylim(0, 2)
# # plt.show()
# fig, ax = plt.subplots()
# y_range = np.linspace(0.001, 0.999, 100)
# for y_i in y_range:
#     comp_q = iast(partial_pressures=[total_f * y_i, total_f * (1-y_i)], isotherms=isotherms, warningoff=True)
#     x_0 = comp_q[0] / np.sum(comp_q)
#     ax.plot(y_i, x_0, 'bo')
# ax.set_xlabel('Gas Phase Mole Fraction of Component 0')
# ax.set_ylabel('Adsorbed Phase Mole Fraction of Component 0')

# for y_i in y_range:
#     comp_q = rast(partial_pressures=[total_f * y_i, total_f * (1-y_i)], isotherms=isotherms, activity_coefficient=ac, warningoff=True)
#     x_0 = comp_q[0] / np.sum(comp_q)
#     ax.plot(y_i, x_0, 'ro')
# ax.legend(['IAST', 'RAST'])
ax.axline((0, 0), slope=1, color='red', linestyle='--')
#print(binary_data['CH4_uptake_absolute[mol/kg]']/(binary_data['CH4_uptake_absolute[mol/kg]'] + binary_data['CO2_uptake_absolute[mol/kg]']))
plt.show()
