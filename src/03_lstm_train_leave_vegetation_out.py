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

# Define subset vegetation types (remove less frequent vegetation types from the analysis)
# This vector will be used to subset the whole raw data for testing purposes, to avoid extra computations
veg_types = ['DBF', 'ENF', 'GRA', 'MF']

# Separate data by vegetation type, for training and testing
data_v, data_other = separate_veg_type(data, args.vegetation, veg_types)

# Create list of sites for cross validation (within chosen vegetation type)
sites = data_v.index.unique()

# Get data dimensions to match LSTM model dimensions
INPUT_FEATURES = data.select_dtypes(include = ['int', 'float']).drop(columns = ['GPP_NT_VUT_REF', 'ai']).shape[1]

# Initialise data.frame to store GPP predictions, from the trained LSTM model
y_pred_sites = {}

# Loop over all sites of chosen vegetation type, 
# An LSTM model is trained on all sites except the "left-out-site"
# for a given number of epochs
for s in sites:
    print(f"Test Site: {s}")

    # TODO: For less common vegetation types, there aren't enough sites to do a proper
    # stratified train-test split, because not all combinations of arid-wet and cold-hot 
    # can be represented in both the train and validation data. A solution needs to be found for this. 

    # Split data (numerical time series and categorical) for leave-site-out cross validation
    # A single site is kept for testing and all others are used for training
    # The sites from other vegetation types are also used to evaluate the generalizability
    # of the model (data_other)
    data_train = data_v.loc[ data_v.index != s ]
    data_test = pd.concat([data_v.loc[ data_v.index == s], data_other])
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
        writer = SummaryWriter(log_dir = f"../model/runs/lstm_lvo_epochs_{args.n_epochs}_patience_{args.patience}_hdim_{args.hidden_dim}/{args.vegetation}_{s}")
    else:
        writer = SummaryWriter(log_dir = f"../model/runs/{args.output_file}/{args.vegetation}_{s}")

    ## Train the model

    # Return best validation R2 score and the center used to normalize training data (repurposed for testing on left-out-site)
    best_r2, train_mean, train_std = train_model(data_train,
                                                 model, optimizer, writer,
                                                 args.n_epochs, args.device, args.patience)
    
    print(f"Validation R2 score for site {s}:  {best_r2}")
    
    # Save model weights from best epoch
    if len(args.output_file)==0:
        torch.save(model,
            f = f"../model/weights/lstm_lvo_epochs_{args.n_epochs}_patience_{args.patience}_hdim_{args.hidden_dim}_{args.vegetation}_{s}.pt")
    else:
        torch.save(model, f = f"../model/weights/{args.output_file}_{args.vegetation}_{s}.pt")

    # Stop logging, for this site
    writer.close()

    ## Model evaluation

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

    print('Prediction output:')
    print(y_pred.shape)

    print(f"R2 score for site {s}: {test_r2}")
    print("")


# Save GPP bias into a data.frame
df_out = pd.read_csv('../data/raw/df_20210510.csv', index_col=0)[['date', 'GPP_NT_VUT_REF']]
df_out = df_out[df_out.index != 'CN-Cng']

for s in df_out.index.unique():
    df_out.loc[[i == s for i in df_out.index], f'gpp_lstm_{args.vegetation}'] = np.asarray(y_pred_sites.get(s))

# Save to a csv, to be processed in R
if len(args.output_file)==0:
    df_out.to_csv(f"../model/preds/lstm_lvo_epochs_{args.n_epochs}_patience_{args.patience}_hdim_{args.hidden_dim}_{args.vegetation}.csv")   
else:
    df_out.to_csv("../model/preds/" + args.output_file)