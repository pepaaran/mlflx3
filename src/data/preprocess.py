# Functions used to preprocess data

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def compute_center(x):

    # Select numeric variables only, without GPP
    x_num = x.select_dtypes(include = ['int', 'float'])
    x_num = x_num.loc[:, ~x_num.columns.str.startswith('GPP')]

    # Calculate mean and standard deviation, per column
    x_mean = x_num.mean()
    x_std = x_num.std()

    return x_mean, x_std

def separate_veg_type(df, v, veg_types):
    """
    A function to select the most common vegetation types (given by 'classid' in the GPP dataset)
    and extract one of the types for model training.

    Args:
        df (DataFrame): Input data containing numerical features and target variable.
        v (str): Vegetation type used for training. It can be included in veg_types or not.
        veg_types (str): Vector of vegetation types used for testing.
    """

    # Select data for training
    df_v = df.loc[df['classid'] == v]

    # Select data for testing, with all selected vegetation types except v
    veg_types.remove(v)
    df_veg_types = df.loc[[any(classid == v for v in veg_types) for classid in df['classid']]]

    return df_v, df_veg_types


# Define a function to normalize data and prepare it for training
def normalize(df,df_test):
    
    # Copy the dataframes to avoid modifying the originals
    result = df.copy()
    result_test=df_test.copy()

    # Normalize each feature by subtracting the mean and dividing by the standard deviation
    for feature_name in df.columns:
        result[feature_name] = (df[feature_name] - df[feature_name].mean()) / df[feature_name].std()
        result_test[feature_name]=(df_test[feature_name]- df[feature_name].mean())/ df[feature_name].std()
    
    return result, result_test


# Group the data by sites and separate time dependented non-time dependented and the target variable
def prepare_df(data, data_test, meta_columns=['classid','igbp_land_use']):

    # Get unique site identifiers from the data
    sites = data.index.unique()
    sites_test = data_test.index.unique()

    # Extract sensor time-dependent data and the target variable (GPP)
    sensor_data = data.drop(columns=['plant_functional_type', 'classid', 'koeppen_code','igbp_land_use', 'GPP_NT_VUT_REF'])
    df_gpp = data['GPP_NT_VUT_REF']
    
    sensor_data_test = data_test.drop(columns=['plant_functional_type', 'classid', 'koeppen_code','igbp_land_use', 'GPP_NT_VUT_REF'])
    df_gpp_test = data_test['GPP_NT_VUT_REF']
    
    # Standardising both training and test data
    df_sensor, df_sensor_test = normalize(sensor_data, sensor_data_test)
    
    # Group the data by site and store each site's data in separate lists
    df_sensor = [df_sensor[df_sensor.index==site] for site in sites if sensor_data[sensor_data.index == site].size != 0 ]
    df_gpp = [df_gpp[df_gpp.index==site] for site in sites]
    
    df_sensor_test = [df_sensor_test[df_sensor_test.index==site] for site in sites_test if sensor_data_test[sensor_data_test.index == site].size != 0 ]
    df_gpp_test = [df_gpp_test[df_gpp_test.index==site] for site in sites_test]

    return df_sensor, df_sensor_test, df_gpp, df_gpp_test