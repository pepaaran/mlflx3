# This script loads the raw dataset, removes unnecessary columns, imputes missing values using K-nearest neighbors
# for specific columns, and then saves the cleaned and imputed dataset to a new CSV file named 'df_imputed.csv'.

# Load dependencies
from sklearn.impute import KNNImputer
import pandas as pd

# Load the raw dataset from a CSV file and remove unnecessary columns
# about site characteristics and net ecosystem fluxes (too correlated with GPP)
data = pd.read_csv('../data/raw/df_20210510.csv', index_col=0).drop(columns=['lat', 'lon', 'elv','c4','whc','LE_F_MDS','NEE_VUT_REF'])

# Remove data for the site 'CN-Cng' due to missing meta-data
data = data[data.index != 'CN-Cng']

sites = data.index.unique()

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

# Save the cleaned and imputed dataset to a new CSV file    
data.to_csv('../data/processed/df_imputed.csv')