# This file contains the cross validation data split
# functions used in the main scripts

# Import dependencies
import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_split_sites(df):
    """
    Function to split the DataFrame into train and validation sets
    based on 'TA_F' mean temperature per 'sitename'.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'TA_F', 'classid', and 'sitename' columns

    Returns:
    - df_train (pd.DataFrame): DataFrame with the train data
    - df_val (pd.DataFrame): DataFrame with the validation data
    """

   # Group by 'sitename' and calculate mean temperature
    grouped = df.groupby('sitename').agg({'TA_F': 'mean', 'classid': 'first', 'ai': 'first'})

    # Discretize numerical columns into bins
    grouped['TA_F_bins'] = pd.cut(grouped['TA_F'], bins=2, labels=False)
    grouped['ai_bins'] = pd.cut(grouped['ai'], bins=2, labels=False)

    # Combine discretized columns into a single categorical column for stratification
    grouped['combined_target'] = grouped['TA_F_bins'].astype(str) + '_' + grouped['ai_bins'].astype(str)


    # Use train_test_split to create two site groups, stratified by mean temperature
    train_df, val_df = train_test_split(grouped, test_size=0.2, 
    stratify=grouped['combined_target'], random_state=42)
    
    # Get train and validation sites
    sites_train = train_df.index
    sites_val = val_df.index

    # Separate the time series data (including the imputed values mask in the last column)
    df_train = df.loc[[any(site == s for s in sites_train) for site in df.index]]
    df_val = df.loc[[any(site == s for s in sites_val) for site in df.index]]

    return df_train, df_val, sites_train, sites_val