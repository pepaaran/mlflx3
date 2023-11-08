import os
import pandas as pd
from torch.utils.data import Dataset
import torch

class gpp_dataset(Dataset):
    def __init__(self, x, x_cat, train_mean, train_std):
        """
        A PyTorch Dataset for GPP prediction.

        Args:
            x (DataFrame): Input data containing numerical features and target variable.
            x_cat (DateFrame): Input 
            train_mean (float): Mean value of training data for centering features.
            train_std (float): Standard deviation of training data for scaling features.
        """
        
        # Select numeric variables only, without GPP
        x_num = x.select_dtypes(include = ['int', 'float'])
        x_num = x_num.loc[:, ~x_num.columns.str.startswith('GPP')]

        # Center data, according to training data center
        x_centered = (x_num - train_mean)/train_std

        # Create tensor for the covariates
        # The pandas DataFrame must be converted to a numpy array
        self.x = torch.tensor(x_centered.values,
                              dtype = torch.float32)
        self.c = torch.tensor(x_cat.values,
                              dtype = torch.float32)
        
        # Define target        
        self.y = torch.tensor(x['GPP_NT_VUT_REF'].values,
                              dtype = torch.float32)

        # Define vector of sites corresponding to the rows in x
        # to be used for indexing
        self.sitename = x.index

        # Define list of unique sites
        self.sites = x.index.unique()

        # Define length of dataset
        # self.len = x.shape[0]         # number of rows
        self.len = len(self.sites)      # number of sites

    def __getitem__(self, idx):
        """
        Get the covariates and target variable for a specific site.

        Args:
            idx (int): Index of the site.

        Returns:
            Thruple of numerical and categorical covariates and target variable for the specified site.
        """
        
        # Select rows corresponding to site idx
        rows = [s == self.sites[idx] for s in self.sitename]
        return self.x[rows], self.y[rows], self.c[rows]
  
    def __len__(self):
        """
        Get the total number of samples (i.e. sites) in the dataset.

        Returns:
            int: The number of samples in the dataset, that is, the number of sites.
        """

        return self.len
    

def compute_center(x):

    # Select numeric variables only
    x_num = x.select_dtypes(include = ['int', 'float'])

    # Calculate mean and standard deviation, per column
    x_mean = x_num.mean()
    x_std = x_num.std()

    return x_mean, x_std


