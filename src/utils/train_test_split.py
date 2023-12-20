# This file contains the cross validation data split
# functions used in the main scripts

# Import dependencies
import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_split_sites(df):
    """
    Function to split the DataFrame into train and validation sets
    based on 'TA_F' mean temperature and 'ai' aridity per 'sitename'.
    20% of data is used for validation.

    Parameters:
    - df (pd.DataFrame): DataFrame containing flux variables used for training and 'TA_F', 'classid', and 'sitename' columns

    Returns:
    - df_train (pd.DataFrame): DataFrame with the train data
    - df_val (pd.DataFrame): DataFrame with the validation data
    """

    # Set test-split size (0-1)
    test_size = 0.2

    # Group by 'sitename' and calculate mean temperature
    grouped = df.groupby('sitename').agg({'TA_F': 'mean', 'classid': 'first', 'ai': 'first'})

    # Discretize numerical columns into bins
    grouped['TA_F_bins'] = pd.cut(grouped['TA_F'], bins=2, labels=False)
    grouped['ai_bins'] = pd.cut(grouped['ai'], bins=2, labels=False)

    # Combine discretized columns into a single categorical column for stratification
    grouped['combined_target'] = grouped['TA_F_bins'].astype(str) + '_' + grouped['ai_bins'].astype(str)
    
    ## Use train_test_split to create two site groups

    # Making sure that there is more than one site in each combined_target group,
    # in order to run train_test_split.
    special_sites = grouped.groupby('combined_target').filter(lambda x: len(x) == 1).index.tolist()
    
    if len(special_sites) == len(grouped.index.unique()):
        # If there are few sites for training (e.g. in leave-vegetation-out)
        # it may lead to all sites being "special", then split at random
        train_df, val_df = train_test_split(grouped, test_size=test_size)

        # Get train and validation sites
        sites_train = train_df.index
        sites_val = val_df.index

    elif len(special_sites):
        # If there is only one site per category, put it in the training set and split the rest
        # stratified by mean temperature and aridity.
        grouped = grouped.drop(special_sites)
        train_df, val_df = train_test_split(grouped, test_size=test_size, stratify=grouped['combined_target'])

        # Get train and validation sites
        sites_train = train_df.index.tolist() + special_sites    # Add weird sites to training set
        sites_val = val_df.index 
    
    else:
        # Use train_test_split to create two site groups, stratified by mean temperature and aridity
        train_df, val_df = train_test_split(grouped, test_size=test_size, stratify=grouped['combined_target'])

        # Get train and validation sites
        sites_train = train_df.index
        sites_val = val_df.index

    # Separate the time series data (including the imputed values mask in the last column)
    df_train = df.loc[[any(site == s for s in sites_train) for site in df.index]]
    df_val = df.loc[[any(site == s for s in sites_val) for site in df.index]]

    return df_train, df_val, sites_train, sites_val