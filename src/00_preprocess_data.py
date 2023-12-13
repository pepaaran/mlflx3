# This script loads the raw dataset, removes unnecessary columns, adds site metadata (aridity index),
# imputes missing values using K-nearest neighbors
# for specific columns, and then saves the cleaned and imputed dataset to a new CSV file named 'df_imputed.csv'.

# Load dependencies
import argparse
from sklearn.impute import KNNImputer
import pandas as pd

# Parse arguments 
parser = argparse.ArgumentParser(description='Data pre-processing')

parser.add_argument('-w', '--wscal', type=int, default=0,
                    help='Keep wscal in the input variables (wscal=0) or remove it (other)')

args = parser.parse_args()

# Load the raw dataset from a CSV file and remove unnecessary columns
# about site characteristics and net ecosystem fluxes (too correlated with GPP)
data = pd.read_csv('../data/raw/df_20210510.csv', index_col=0).drop(columns=['lat', 'lon', 'elv','c4','whc','LE_F_MDS','NEE_VUT_REF'])

if args.wscal:
    # Remove wscal (computed by SPLASH), which accounts for the memory of the water balance
    data = data.drop(columns = ['wscal'])
    print("Removing wscal from the variables used for modelling")

# Read metadata from FLUXNET sites to obtain aridity index (ai)
# This file was provided by Beni and includes only 53 sites, in the future it may
# contain more variables to extend the flux data, but the pipeline to obtain
# the site characteristics should be written transparently.
df_meta = pd.read_csv("../data/external/fluxnet2015_sites_metainfo.csv", index_col = 0)
df_meta.set_index('mysitename', inplace=True)

# Remove data for the site 'CN-Cng' due to missing meta-data
data = data[data.index != 'CN-Cng']

sites = data.index.unique()

print("Imputing temperature (day and night) and GPP values")

# Impute 'TA_F_DAY' and 'TA_F_NIGHT' columns using 'TA_F' and 'SW_IN_F'
# Iterate over sites to perform imputation for each site
df =  data[['TA_F','SW_IN_F','TA_F_DAY', 'TA_F_NIGHT']]
for s in sites:
    impute = KNNImputer()
    x = df[df.index == s].values
    x = impute.fit_transform(x)
    data.loc[data.index == s, 'TA_F_DAY'] = x[:,2]
    data.loc[data.index == s, 'TA_F_NIGHT'] = x[:,3]

# Impute 'GPP_NT_VUT_REF' column using multiple, selected features
# Iterate over sites to perform imputation for each site
df =  data[['TA_F','SW_IN_F','TA_F_DAY', 'LW_IN_F','WS_F','P_F', 'VPD_F', 'GPP_NT_VUT_REF']]
for s in sites:
    impute = KNNImputer()
    x = df[df.index == s].values
    x = impute.fit_transform(x)
    data.loc[data.index == s, 'GPP_NT_VUT_REF'] = x[:,-1]

print("Merging aridity index from FLUXNET metadata")

# Merge the data with the metadata based on their indices
# The aridity index will be used for the stratified train-test splits
data = pd.merge(data, df_meta[['ai']], left_on='sitename', right_index=True, how='left')

# Add a column indicating whether the GPP values were original (True) or imputed (False)
# to be used as a mask in the model testing
data['not_imputed'] = ~df['GPP_NT_VUT_REF'].isna()

# Save the cleaned and imputed dataset to a new CSV file    
data.to_csv('../data/processed/df_imputed.csv')
print("Imputed data saved to data/processed/df_imputed.csv")