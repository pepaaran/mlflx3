# LSTM model with leave-site-out cross-validation on a single vegetation type
# This script extracts the data for DBF, ENF, GRA and MF vegetation types
# and trains the LSTM model on sites from a given vegetation type at a time.
# Lastly, the model is trained with data from sites with these 4 vegetation types.

# Custom modules and functions
from models.lstm_model import Model
from data.preprocess import separate_veg_type
from data.dataloader import gpp_dataset
from utils.utils import set_seed
from utils.train_test_loops import *
from utils.train_model import train_model

# Load necessary dependencies
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch


# Parse arguments 
parser = argparse.ArgumentParser(description='CV LSTM')

parser.add_argument('-v', '--vegetation', type=str, required=True,
                    help='String indicating the vegetation type used for training (DBF, ENF, GRA or MF)')

parser.add_argument('-device', '--device', default='cuda:0' ,type=str,
                      help='Indices of GPU to enable')

parser.add_argument('-e', '--n_epochs', default=150, type=int,
                      help='Number of training epochs (per site, for the leave-site-out CV)')

parser.add_argument('-o', '--output_file', default='', type=str,
                    help='File name to save output')

parser.add_argument('-d', '--hidden_dim', default=256, type=int,
                    help='Hidden dimension of the LSTM model')

parser.add_argument('-p', '--patience', default=10, type=int,
                    help='Number of iterations (patience threshold) used for early stopping')

args = parser.parse_args()


# Set random seeds for reproducibility
set_seed(40)

print("Starting leave-vegetation-out on LSTM model:")
print(f"> Left out vegetation type: {args.vegetation}")
print(f"> Device: {args.device}")
print(f"> Epochs: {args.n_epochs}")
print(f"> Early stopping after {args.patience} epochs without improvement")
print(f"Hidden dimension of LSTM model: {args.hidden_dim}")

# Read imputed data
data = pd.read_csv('../data/processed/df_imputed.csv', index_col=0)

# Read raw data to compute bias
df_out = pd.read_csv('../data/raw/df_20210510.csv', index_col=0)[['date', 'GPP_NT_VUT_REF']]
df_out = df_out[df_out.index != 'CN-Cng']

# Define subset vegetation types (remove less frequent vegetation types from the analysis)
# This vector will be used to subset the whole raw data for testing purposes, to avoid extra computations
veg_types = ['DBF', 'ENF', 'GRA', 'MF']

# Separate data by vegetation type, for training and testing
data_v, data_other = separate_veg_type(data, args.vegetation, veg_types)

# Create list of sites for cross validation (within chosen vegetation type) and for bias evaluation
sites = data_v.index.unique()
sites_other = data_other.index.unique()

# Get data dimensions to match LSTM model dimensions
INPUT_FEATURES = data.select_dtypes(include = ['int', 'float']).drop(columns = ['GPP_NT_VUT_REF', 'ai']).shape[1]

# Initialise data.frame to store GPP predictions and bias, from the trained LSTM model
y_pred_sites = {}
bias = {}
for s_out in sites_other:
    bias[s_out] = []

# Loop over all sites of chosen vegetation type, 
# An LSTM model is trained on all sites except the "left-out-site"
# for a given number of epochs
for s in [sites[1]]:
    print(f"Test Site: {s}")

    # TODO: For less common vegetation types, there aren't enough sites to do a proper
    # stratified train-test split, because not all combinations of arid-wet and cold-hot 
    # can be represented in both the train and validation data. A solution needs to be found for this. 

    # Split data (numerical time series and categorical) for leave-site-out cross validation
    # A single site is kept for testing and all others are used for training
    # The sites from other vegetation types are also used to evaluate the generalizability
    # of the model (data_other)
    data_train = data_v.loc[ data_v.index != s ]
    # data_test = pd.concat([data_v.loc[ data_v.index == s], data_other])       # old idea
    data_test = data_v.loc[ data_v.index == s]
    print(data_train.index.unique())
    ## Define model to be trained

    # Initialise the LSTM model, set layer dimensions to match data
    model = Model(input_dim = INPUT_FEATURES, 
                    hidden_dim = args.hidden_dim,
                    num_layers = 1).to(device = args.device)

    # Initialise the optimiser
    optimizer = torch.optim.Adam(model.parameters())

    # Initiate tensorboard logging instance for this site
    if len(args.output_file) == 0:
        writer = SummaryWriter(log_dir = f"../models/runs/lstm_lvo_epochs_{args.n_epochs}_patience_{args.patience}_hdim_{args.hidden_dim}/{args.vegetation}_{s}")
    else:
        writer = SummaryWriter(log_dir = f"../models/runs/{args.output_file}/{args.vegetation}_{s}")

    ## Train the model

    # Return best validation R2 score and the center used to normalize training data (repurposed for testing on left-out-site)
    best_r2, train_mean, train_std = train_model(data_train,
                                                 model, optimizer, writer,
                                                 args.n_epochs, args.device, args.patience)
    
    print(f"Validation R2 score for site {s}:  {best_r2}")
    
    # Save model weights from best epoch
    if len(args.output_file)==0:
        torch.save(model,
            f = f"../models/weights/lstm_lvo_epochs_{args.n_epochs}_patience_{args.patience}_hdim_{args.hidden_dim}_{args.vegetation}_{s}.pt")
    else:
        torch.save(model, f = f"../models/weights/{args.output_file}_{args.vegetation}_{s}.pt")

    # Stop logging, for this site
    writer.close()

    ## Model evaluation on leaf-out site

    # Format test pytorch dataset for the data loader
    test_ds = gpp_dataset(data_test, train_mean, train_std)

    # Run data loader with batch_size = 1
    # Due to different time series lengths per site,
    # we cannot load several sites per batch
    test_dl = DataLoader(test_ds, batch_size = 1, shuffle = True)

    # Evaluate model on test set, removing imputed GPP values
    test_loss, test_r2, y_pred = test_loop(test_dl, model, args.device)

    # Save prediction for the left-out site
    y_pred_sites[s] = y_pred

    print(f"R2 score for site {s}: {test_r2}")
    print("")

    # Compute prediction bias
    bias[s] = np.mean(y_pred - df_out.loc[ df_out.index == s]['GPP_NT_VUT_REF'])

    print(f"Prediction bias for site {s}: {bias[s]}")
    print("")

    ## Model evaluation on other vegetations
    
    for s_other in sites_other:
        # Format test pytorch dataset for the data loader, using data from onte site only
        test_ds_other = gpp_dataset(data_other.loc[ data_other.index == s_other], 
                                    train_mean, train_std)

        # Run data loader with batch_size = 1
        # Due to different time series lengths per site,
        # we cannot load several sites per batch
        test_dl = DataLoader(test_ds_other, batch_size = 1, shuffle = True)

        # Evaluate model on test set, removing imputed GPP values
        test_loss_other, test_r2_other, y_pred_other = test_loop(test_dl, model, args.device)

        # Compute prediction bias, save to aggregate later
        bias[s_other].append( np.mean(y_pred_other - df_out.loc[ df_out.index == s_other]['GPP_NT_VUT_REF']) )


# Save GPP bias into a data.frame, averaging over all trained models for the sites of other veg types
df_bias = pd.DataFrame({site:np.mean(b) for site,b in bias.items()}.items(),
                       columns = ['sitename', 'bias']
                       ).set_index('sitename')

# Save to a csv, to be processed in R
if len(args.output_file)==0:
    df_bias.to_csv(f"../models/preds/bias_lstm_lvo_epochs_{args.n_epochs}_patience_{args.patience}_hdim_{args.hidden_dim}_{args.vegetation}.csv")   
else:
    df_bias.to_csv("../models/preds/bias_" + args.output_file) 

# Save GPP predictions into data.frame
for s in sites:
    df_out.loc[[i == s for i in df_out.index], f'gpp_lstm_{args.vegetation}'] = np.asarray(y_pred_sites.get(s))

# Save to a csv, to be processed in R
if len(args.output_file)==0:
    df_out.to_csv(f"../models/preds/lstm_lvo_epochs_{args.n_epochs}_patience_{args.patience}_hdim_{args.hidden_dim}_{args.vegetation}.csv")   
else:
    df_out.to_csv("../models/preds/" + args.output_file)