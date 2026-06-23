from pathlib import Path

import numpy as np
import pandas as pd
from numpy.dtypes import StringDType

from pyrast.calculations.iast import iast
from pyrast.isotherms import (
    InterpolatorIsotherm,
    ModelIsotherm,
    PCHIPInterpolatorIsotherm,
)


def build_mixture_df(path: str = '/Users/jonah/Desktop/Research/IAST Project/pyRAST/tests/test_inputs/Results/', temp: str = '298K') -> pd.DataFrame:
    mof_list = []
    mixture_list = []

    for d in Path(path).iterdir():
        if d.is_dir() and not d.name.startswith('.'):
            mof_list.append(d.name)
    for mof in mof_list:
        # get single components allowed
        component_list = []
        for file in Path(f'{path}{mof}/').iterdir():
            if len(str(file.name).split('_')) == 4 and str(file.name).split('_')[2] == temp:
                component_list.append(str(file.name).split('_')[1])
        for file in Path(f'{path}{mof}/').iterdir():
            if len(str(file.name).split('_')) == 7 and str(file.name).split('_')[5] == temp:
                component1 = str(file.name).split('_')[1]
                component2 = str(file.name).split('_')[2]
                frac1 = float(str(file.name).split('_')[3])/100
                frac2 = float(str(file.name).split('_')[4])/100
                if component1 in component_list and component2 in component_list:
                    mixture_list.append({'MOF': mof,
                                                    'Component1': component1, 
                                                    'Component2': component2, 
                                                    'MoleFrac1': frac1, 
                                                    'MoleFrac2': frac2, 
                                                    'Temp[K]': temp})
    header = ['MOF','Component1', 'Component2',
              'MoleFrac1', 'MoleFrac2', 'Temp[K]']
    mixture_df = pd.DataFrame(mixture_list, columns=header)

    return mixture_df.sort_values(by=['MOF', 'Component1', 'Component2', 
                                      'MoleFrac1'], ignore_index=True)

def get_binary_loadings(df: pd.DataFrame) -> pd.DataFrame:
    loading_list = []
    for _, row in df.iterrows():
        mof = row['MOF']
        component1 = row['Component1']
        component2 = row['Component2']
        frac1 = row['MoleFrac1']*100
        frac2 = row['MoleFrac2']*100
        temp = row['Temp[K]']
        file_name = (f'/Users/jonah/Desktop/Research/IAST Project/pyRAST/tests/test_inputs/Results/{mof}/{mof}_{component1}_{component2}_{frac1}'
                     f'_{frac2}_{temp}_isotherm.csv')
        isotherm_df = pd.read_csv(file_name)
        comp1load = list(isotherm_df[f'{component1}_uptake_absolute[mol/kg]'])
        comp1load_unc = list(isotherm_df[f'{component1}_uptake_absolute_uncertainty[mol/kg]'])
        comp2load = list(isotherm_df[f'{component2}_uptake_absolute[mol/kg]'])
        comp2load_unc = list(isotherm_df[f'{component2}_uptake_absolute_uncertainty[mol/kg]'])
        fug1 = list(isotherm_df[f'{component1}_fugacity[Pa]'])
        fug2 = list(isotherm_df[f'{component2}_fugacity[Pa]'])
        pressure = list(isotherm_df['total_pressure[Pa]'])
        totalfug = [f1 + f2 for f1, f2 in zip(fug1, fug2)]
        loading_list.append([pressure, comp1load, comp1load_unc, comp2load,
                             comp2load_unc, fug1, fug2, totalfug])
    header = ['Pressure[Pa]', 'C1Loading[mol/kg]', 'C1LoadingUnc[mol/kg]',
              'C2Loading[mol/kg]', 'C2LoadingUnc[mol/kg]',
              'Fugacity1[Pa]', 'Fugacity2[Pa]', 'TotalFugacity[Pa]']
    loading_df = pd.DataFrame(loading_list, columns=header)
    return pd.concat([df, loading_df], axis=1).explode(header)

def get_single_loadings(mof: str, component: str, temp: str = '298K') -> pd.DataFrame:
    file_name = f'/Users/jonah/Desktop/Research/IAST Project/pyRAST/tests/test_inputs/Results/{mof}/{mof}_{component}_{temp}_isotherm.csv'
    isotherm_df = pd.read_csv(file_name)
    return isotherm_df

def calculate_selectivity(data: pd.DataFrame) -> pd.DataFrame:
    selectivity = np.zeros(len(data))
    uncertainty = np.zeros(len(data))
    higher_load = np.empty(len(data), dtype=StringDType())
    c=0
    for _,row in data.iterrows():
        comp1load = row['C1Loading[mol/kg]']
        comp2load = row['C2Loading[mol/kg]']
        c1unc = row['C1LoadingUnc[mol/kg]']
        c2unc = row['C2LoadingUnc[mol/kg]']
        frac1 = row['MoleFrac1']
        frac2 = row['MoleFrac2']
        if comp1load == 0 or comp2load == 0:
            selectivity[c] = np.nan
            uncertainty[c] = np.nan
            higher_load[c] = np.nan
        else:
            selectivity[c] = (comp1load / comp2load)/(frac1/frac2)
            higher_load[c] = row['Component1']
            if selectivity[c] < 1:
                selectivity[c] = 1/selectivity[c]
                higher_load[c] = row['Component2']
            uncertainty[c] = selectivity[c] * np.sqrt((c1unc/comp1load)**2 + (c2unc/comp2load)**2)
        c+=1
    data['Selectivity'] = selectivity
    data['SelectivityUnc'] = uncertainty
    data['PreferredComponent'] = higher_load
    return data

def fit_isotherms(input_data: pd.DataFrame, temp: str = '298K', min_index: int = 0,
                  max_index: int = 10):
    mofs = input_data['MOF'].unique()
    failures = 0
    for _, mof in enumerate(mofs):
        # get single component isotherms
        butane = get_single_loadings(mof, 'butane', temp).iloc[min_index:max_index]
        ch4 = get_single_loadings(mof, 'CH4', temp).iloc[min_index:max_index]
        co2 = get_single_loadings(mof, 'CO2', temp).iloc[min_index:max_index]
        n2 = get_single_loadings(mof, 'N2', temp).iloc[min_index:max_index]

        # fit isotherms to get parameters
        try:
            butane_isotherm = ModelIsotherm(df=butane, 
                                                        loading_key='butane_uptake_absolute[mol/kg]',
                                                        pressure_key='butane_fugacity[Pa]',
                                                        model='DSLangmuir',
                                                        optimization_mode='enhanced')
        except Exception as e:
            failures += 1
            print(e)
        try:
             ch4_isotherm = InterpolatorIsotherm(df=ch4, 
                                                        loading_key='CH4_uptake_absolute[mol/kg]',
                                                        pressure_key='CH4_fugacity[Pa]',
                                                        optimization_mode='enhanced')
        except Exception as e:
            failures += 1
            print(e)
        try:
            co2_isotherm = ModelIsotherm(df=co2, model='DSLangmuir',
                                                   loading_key='CO2_uptake_absolute[mol/kg]',
                                                   pressure_key='CO2_fugacity[Pa]',
                                                   optimization_mode='enhanced')
        except Exception as e:
            failures += 1
            print(e)
        try:
            n2_isotherm = ModelIsotherm(df=n2, model='DSLangmuir',
                                                  loading_key='N2_uptake_absolute[mol/kg]',
                                                  pressure_key='N2_fugacity[Pa]',
                                                  optimization_mode='enhanced')
        except Exception as e:
            failures += 1
            print(e)
    print(f'Fitting complete with {failures} failures.')

def calculate_iast_langmuir(input_data: pd.DataFrame,
                                temp: str, min_index: int = 0,
                                max_index: int = 10, sparse: bool = False) -> pd.DataFrame:
    mofs = input_data['MOF'].unique()
    c1_loading_iast = np.zeros(len(input_data))
    c2_loading_iast = np.zeros(len(input_data))
    selectivity_iast = np.zeros(len(input_data))
    i = 0
    # want to work with 1 mof at a time to reduce fitting each time
    for _, mof in enumerate(mofs):
        mof_data = input_data[input_data['MOF'] == mof]
        # get single component isotherms
        butane = get_single_loadings(mof, 'butane', temp).iloc[min_index:max_index]
        ch4 = get_single_loadings(mof, 'CH4', temp).iloc[min_index:max_index]
        co2 = get_single_loadings(mof, 'CO2', temp).iloc[min_index:max_index]
        n2 = get_single_loadings(mof, 'N2', temp).iloc[min_index:max_index]

        if sparse:
            butane = get_single_loadings(mof, 'butane', temp).iloc[::2]
            ch4 = get_single_loadings(mof, 'CH4', temp).iloc[::2]
            co2 = get_single_loadings(mof, 'CO2', temp).iloc[::2]
            n2 = get_single_loadings(mof, 'N2', temp).iloc[::2]

        # fit isotherms to get parameters for IAST
        try:
            butane_isotherm = ModelIsotherm(df=butane, model='Langmuir',
                                                        loading_key='butane_uptake_absolute[mol/kg]',
                                                        pressure_key='butane_fugacity[Pa]',
                                                        optimization_mode='enhanced')
        except Exception as e:
            print(e)
            butane_isotherm = None
        try:
            ch4_isotherm = ModelIsotherm(df=ch4, model='Langmuir',
                                                 loading_key='CH4_uptake_absolute[mol/kg]',
                                                 pressure_key='CH4_fugacity[Pa]',
                                                 optimization_mode='enhanced')
        except Exception as e:
            print(e)
            ch4_isotherm = None
        try:
            co2_isotherm = ModelIsotherm(df=co2, model='Langmuir',
                                                 loading_key='CO2_uptake_absolute[mol/kg]',
                                                 pressure_key='CO2_fugacity[Pa]',
                                                 optimization_mode='enhanced')
        except Exception as e:
            print(e)
            co2_isotherm = None
        try:
            n2_isotherm = ModelIsotherm(df=n2, model='Langmuir',
                                                loading_key='N2_uptake_absolute[mol/kg]',
                                                pressure_key='N2_fugacity[Pa]',
                                                optimization_mode='enhanced')
        except Exception as e:
            print(e)
            n2_isotherm = None

        single_isotherms = {'butane': butane_isotherm, 'CH4': ch4_isotherm,
                            'CO2': co2_isotherm, 'N2': n2_isotherm}
        # for each mixture, calculate binary loadings and selectivity
        for _, row in mof_data.iterrows():
            component1 = row['Component1']
            component2 = row['Component2']
            fug1 = row['Fugacity1[Pa]']
            fug2 = row['Fugacity2[Pa]']
            molefrac1 = row['MoleFrac1']
            molefrac2 = row['MoleFrac2']

            isotherms = [single_isotherms[component1], single_isotherms[component2]]
            fugacities = [fug1, fug2]
            try:
                loadings = iast(fugacities, isotherms)
            except Exception as e:
                print(e)
                loadings = [np.nan, np.nan]
            c1_loading_iast[i] = loadings[0]
            c2_loading_iast[i] = loadings[1]

            selectivity_iast[i] = (c1_loading_iast[i] / c2_loading_iast[i]) / (molefrac1 / molefrac2)
            if selectivity_iast[i] < 1:
                selectivity_iast[i] = 1 / selectivity_iast[i]
            i+=1

    input_data['C1LoadingIAST[mol/kg]'] = c1_loading_iast
    input_data['C2LoadingIAST[mol/kg]'] = c2_loading_iast
    input_data['SelectivityIAST'] = selectivity_iast

    percent_error = 100 * np.abs(input_data['SelectivityIAST'] - input_data['Selectivity']) / input_data['Selectivity']
    input_data['PercentErrorIAST'] = percent_error
    return input_data

def calculate_iast_interpolator(input_data: pd.DataFrame,
                                temp: str, min_index: int,
                                max_index: int) -> pd.DataFrame:
    mofs = input_data['MOF'].unique()
    c1_loading_iast = np.zeros(len(input_data))
    c2_loading_iast = np.zeros(len(input_data))
    selectivity_iast = np.zeros(len(input_data))
    i = 0
    # want to work with 1 mof at a time to reduce fitting each time
    for _, mof in enumerate(mofs):
        print(mof)
        mof_data = input_data[input_data['MOF'] == mof]
        # get single component isotherms
        butane = get_single_loadings(mof, 'butane', temp).iloc[min_index:max_index]
        ch4 = get_single_loadings(mof, 'CH4', temp).iloc[min_index:max_index]
        co2 = get_single_loadings(mof, 'CO2', temp).iloc[min_index:max_index]
        n2 = get_single_loadings(mof, 'N2', temp).iloc[min_index:max_index]

        # fit isotherms to get parameters for IAST
        butane_isotherm = PCHIPInterpolatorIsotherm(butane,
                                                      loading_key='butane_uptake_absolute[mol/kg]',
                                                      pressure_key='butane_fugacity[Pa]')
        ch4_isotherm = PCHIPInterpolatorIsotherm(ch4,
                                                   loading_key='CH4_uptake_absolute[mol/kg]',
                                                   pressure_key='CH4_fugacity[Pa]')
        co2_isotherm = PCHIPInterpolatorIsotherm(co2,
                                                   loading_key='CO2_uptake_absolute[mol/kg]',
                                                   pressure_key='CO2_fugacity[Pa]')
        n2_isotherm = PCHIPInterpolatorIsotherm(n2,
                                                  loading_key='N2_uptake_absolute[mol/kg]',
                                                  pressure_key='N2_fugacity[Pa]')
        single_isotherms = {'butane': butane_isotherm, 'CH4': ch4_isotherm,
                            'CO2': co2_isotherm, 'N2': n2_isotherm}
        # for each mixture, calculate binary loadings and selectivity
        for _, row in mof_data.iterrows():
            component1 = row['Component1']
            component2 = row['Component2']
            fug1 = row['Fugacity1[Pa]']
            fug2 = row['Fugacity2[Pa]']
            molefrac1 = row['MoleFrac1']
            molefrac2 = row['MoleFrac2']

            isotherms = [single_isotherms[component1], single_isotherms[component2]]
            fugacities = [fug1, fug2]
            try:
                loadings = iast(fugacities, isotherms)
            except Exception as e:
                print(e)
                loadings = [np.nan, np.nan]
            c1_loading_iast[i] = loadings[0]
            c2_loading_iast[i] = loadings[1]

            selectivity_iast[i] = (c1_loading_iast[i] / c2_loading_iast[i]) / (molefrac1 / molefrac2)
            if selectivity_iast[i] < 1:
                selectivity_iast[i] = 1 / selectivity_iast[i]
            i+=1

    input_data['C1LoadingIAST[mol/kg]'] = c1_loading_iast
    input_data['C2LoadingIAST[mol/kg]'] = c2_loading_iast
    input_data['SelectivityIAST'] = selectivity_iast

    percent_error = 100 * np.abs(input_data['SelectivityIAST'] - input_data['Selectivity']) / input_data['Selectivity']
    input_data['PercentErrorIAST'] = percent_error
    return input_data

def calculate_iast_interpolator_langmuir(input_data: pd.DataFrame,
                                temp: str, min_index: int,
                                max_index: int) -> pd.DataFrame:
    mofs = input_data['MOF'].unique()
    c1_loading_iast = np.zeros(len(input_data))
    c2_loading_iast = np.zeros(len(input_data))
    selectivity_iast = np.zeros(len(input_data))
    i = 0
    # want to work with 1 mof at a time to reduce fitting each time
    for _, mof in enumerate(mofs):
        mof_data = input_data[input_data['MOF'] == mof]
        # get single component isotherms
        butane = get_single_loadings(mof, 'butane', temp).iloc[min_index:max_index]
        ch4 = get_single_loadings(mof, 'CH4', temp).iloc[min_index:max_index]
        co2 = get_single_loadings(mof, 'CO2', temp).iloc[min_index:max_index]
        n2 = get_single_loadings(mof, 'N2', temp).iloc[min_index:max_index]

        # fit isotherms to get parameters for IAST
        butane_isotherm = InterpolatorIsotherm(butane,
                                                      loading_key='butane_uptake_absolute[mol/kg]',
                                                      pressure_key='butane_fugacity[Pa]',
                                                      extrap_method='Langmuir',
                                                      optimization_mode='enhanced')
        ch4_isotherm = InterpolatorIsotherm(ch4,
                                                   loading_key='CH4_uptake_absolute[mol/kg]',
                                                   pressure_key='CH4_fugacity[Pa]',
                                                   extrap_method='Langmuir',
                                                   optimization_mode='enhanced')
        co2_isotherm = InterpolatorIsotherm(co2,
                                                   loading_key='CO2_uptake_absolute[mol/kg]',
                                                   pressure_key='CO2_fugacity[Pa]',
                                                   extrap_method='Langmuir',
                                                   optimization_mode='enhanced')
        n2_isotherm = InterpolatorIsotherm(n2,
                                                  loading_key='N2_uptake_absolute[mol/kg]',
                                                  pressure_key='N2_fugacity[Pa]',
                                                  extrap_method='Langmuir',
                                                  optimization_mode='enhanced')
        single_isotherms = {'butane': butane_isotherm, 'CH4': ch4_isotherm,
                            'CO2': co2_isotherm, 'N2': n2_isotherm}
        # for each mixture, calculate binary loadings and selectivity
        for _, row in mof_data.iterrows():
            component1 = row['Component1']
            component2 = row['Component2']
            fug1 = row['Fugacity1[Pa]']
            fug2 = row['Fugacity2[Pa]']
            molefrac1 = row['MoleFrac1']
            molefrac2 = row['MoleFrac2']

            isotherms = [single_isotherms[component1], single_isotherms[component2]]
            fugacities = [fug1, fug2]
            try:
                loadings = iast(fugacities, isotherms)
            except Exception as e:
                print(e)
                loadings = [np.nan, np.nan]
            c1_loading_iast[i] = loadings[0]
            c2_loading_iast[i] = loadings[1]

            selectivity_iast[i] = (c1_loading_iast[i] / c2_loading_iast[i]) / (molefrac1 / molefrac2)
            if selectivity_iast[i] < 1:
                selectivity_iast[i] = 1 / selectivity_iast[i]
            i+=1

    input_data['C1LoadingIAST[mol/kg]'] = c1_loading_iast
    input_data['C2LoadingIAST[mol/kg]'] = c2_loading_iast
    input_data['SelectivityIAST'] = selectivity_iast

    percent_error = 100 * np.abs(input_data['SelectivityIAST'] - input_data['Selectivity']) / input_data['Selectivity']
    input_data['PercentErrorIAST'] = percent_error
    return input_data

def calculate_iast_interpolator_linear(input_data: pd.DataFrame,
                                temp: str, min_index: int,
                                max_index: int) -> pd.DataFrame:
    mofs = input_data['MOF'].unique()
    c1_loading_iast = np.zeros(len(input_data))
    c2_loading_iast = np.zeros(len(input_data))
    selectivity_iast = np.zeros(len(input_data))
    i = 0
    # want to work with 1 mof at a time to reduce fitting each time
    for _, mof in enumerate(mofs):
        mof_data = input_data[input_data['MOF'] == mof]
        # get single component isotherms
        butane = get_single_loadings(mof, 'butane', temp).iloc[min_index:max_index]
        ch4 = get_single_loadings(mof, 'CH4', temp).iloc[min_index:max_index]
        co2 = get_single_loadings(mof, 'CO2', temp).iloc[min_index:max_index]
        n2 = get_single_loadings(mof, 'N2', temp).iloc[min_index:max_index]

        # fit isotherms to get parameters for IAST
        butane_isotherm = InterpolatorIsotherm(butane,
                                                      loading_key='butane_uptake_absolute[mol/kg]',
                                                      pressure_key='butane_fugacity[Pa]',
                                                      extrap_method='linear')
        ch4_isotherm = InterpolatorIsotherm(ch4,
                                                   loading_key='CH4_uptake_absolute[mol/kg]',
                                                   pressure_key='CH4_fugacity[Pa]',
                                                   extrap_method='linear')
        co2_isotherm = InterpolatorIsotherm(co2,
                                                   loading_key='CO2_uptake_absolute[mol/kg]',
                                                   pressure_key='CO2_fugacity[Pa]',
                                                   extrap_method='linear')
        n2_isotherm = InterpolatorIsotherm(n2,
                                                  loading_key='N2_uptake_absolute[mol/kg]',
                                                  pressure_key='N2_fugacity[Pa]',
                                                  extrap_method='linear')
        single_isotherms = {'butane': butane_isotherm, 'CH4': ch4_isotherm,
                            'CO2': co2_isotherm, 'N2': n2_isotherm}
        # for each mixture, calculate binary loadings and selectivity
        for _, row in mof_data.iterrows():
            component1 = row['Component1']
            component2 = row['Component2']
            fug1 = row['Fugacity1[Pa]']
            fug2 = row['Fugacity2[Pa]']
            molefrac1 = row['MoleFrac1']
            molefrac2 = row['MoleFrac2']

            isotherms = [single_isotherms[component1], single_isotherms[component2]]
            fugacities = [fug1, fug2]
            try:
                loadings = iast(fugacities, isotherms)
            except Exception as e:
                print(e)
                loadings = [np.nan, np.nan]
            c1_loading_iast[i] = loadings[0]
            c2_loading_iast[i] = loadings[1]

            selectivity_iast[i] = (c1_loading_iast[i] / c2_loading_iast[i]) / (molefrac1 / molefrac2)
            if selectivity_iast[i] < 1:
                selectivity_iast[i] = 1 / selectivity_iast[i]
            i+=1

    input_data['C1LoadingIAST[mol/kg]'] = c1_loading_iast
    input_data['C2LoadingIAST[mol/kg]'] = c2_loading_iast
    input_data['SelectivityIAST'] = selectivity_iast

    percent_error = 100 * np.abs(input_data['SelectivityIAST'] - input_data['Selectivity']) / input_data['Selectivity']
    input_data['PercentErrorIAST'] = percent_error
    return input_data

#binary_mixture_df = build_mixture_df()
#binary_mixture_df = get_binary_loadings(binary_mixture_df)
#fit_isotherms(binary_mixture_df, temp='298K', min_index=0, max_index=8)
#binary_mixture_df = calculate_selectivity(binary_mixture_df)
#binary_mixture_df = calculate_iast_langmuir(binary_mixture_df, temp='298K', min_index=0, max_index=8)
#binary_mixture_df.to_csv('/Users/jonah/Desktop/Research/IAST Project/pyRAST/tests/isotherms/binary_mixture_iast.csv', index=False)

binary_mixture_df = build_mixture_df()
binary_mixture_df = get_binary_loadings(binary_mixture_df)
binary_mixture_df = calculate_selectivity(binary_mixture_df)
binary_mixture_df = calculate_iast_interpolator(binary_mixture_df, temp='298K', min_index=0, max_index=10)
binary_mixture_df.to_csv('/Users/jonah/Desktop/Research/IAST Project/pyRAST/tests/isotherms/binary_mixture_iast_interpolator.csv', index=False)
