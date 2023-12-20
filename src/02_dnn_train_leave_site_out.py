# This is the final DNN model with leave-one-site-out cross-validation

# Custom modules and functions
from models.dnn_model import Model
from data.preprocess import compute_center
from data.dataloader import gpp_dataset
from utils.utils import set_seed
from utils.train_test_loops import train_loop, test_loop
from utils.train_model import train_model

# Load necessary dependencies
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch


# Parse arguments 
parser = argparse.ArgumentParser(description='CV DNN')

parser.add_argument('-device', '--device', default='cuda:0' ,type=str,
                      help='Indices of GPU to enable')

parser.add_argument('-e', '--n_epochs', default=150, type=int,
                      help='Number of training epochs (per site, for the leave-site-out CV)')

parser.add_argument('-o', '--output_file', default='', type=str,
                    help='File name to save output')

parser.add_argument('-p', '--patience', default=10, type=int,
                    help='Number of iterations (patience threshold) used for early stopping')

args = parser.parse_args()


# Set random seeds for reproducibility
set_seed(40)

print("Starting leave-site-out on DNN model:")
print(f"> Device: {args.device}")
print(f"> Epochs: {args.n_epochs}")
print(f"> Early stopping after {args.patience} epochs without improvement")

# Read imputed and raw data# Read imputed data, including variables for stratified train-test split and imputation flag
data = pd.read_csv('../data/processed/df_imputed.csv', index_col=0)

# Create list of sites for leave-site-out cross validation
sites = data.index.unique()

# Get data dimensions to match LSTM model dimensions
INPUT_FEATURES = data.select_dtypes(include = ['int', 'float']).drop(columns = ['GPP_NT_VUT_REF', 'ai']).shape[1]

# Initialise data.frame to store GPP predictions, from the trained LSTM model
y_pred_sites = {}

# Loop over all sites, 
# An LSTM model is trained on all sites except the "left-out-site"
# for a given number of epochs
for s in sites:

    # Split data (numerical time series) for leave-site-out cross validation
    # A single site is kept for testing and all others are used for training
    data_train = data.loc[ data.index != s ]
    data_test = data.loc[ data.index == s]

    ## Define the model to be trained

    # Initialise the DNN model, set layer dimensions to match data
    model = Model(input_dim = INPUT_FEATURES).to(device = args.device)

    # Initialise the optimiser
    optimizer = torch.optim.Adam(model.parameters())

    # Initiate tensorboard logging instance for this site
    if len(args.output_file) == 0:
        writer = SummaryWriter(log_dir = f"../models/runs/dnn_lso_epochs_{args.n_epochs}_patience_{args.patience}/{s}")
    else:
        writer = SummaryWriter(log_dir = f"../models/runs/{args.output_file}/{s}")


    ## Train the model

    # Return best validation R2 score and the center used to normalize training data (repurposed for testing on leaf-out site)
    best_r2, train_mean, train_std = train_model(data_train,
                                                 model, optimizer, writer,
                                                 args.n_epochs, args.device, args.patience)
    
    print(f"Validation R2 score for site {s}:  {best_r2}")

    # Save model weights from best epoch
    if len(args.output_file)==0:
        torch.save(model,
            f = f"../models/weights/dnn_lso_epochs_{args.n_epochs}_patience_{args.patience}_{s}.pt")
    else:
        torch.save(model, f = f"../models/weights/{args.output_file}_{s}.pt")

    # Stop logging, for this site
    writer.close()


    ## Model evaluation

    # Format pytorch dataset for the data loader
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

    

# Save predictions into a data.frame. aligning with raw data
df_out = pd.read_csv('../data/raw/df_20210510.csv', index_col=0)[['date', 'GPP_NT_VUT_REF']]
df_out = df_out[df_out.index != 'CN-Cng']

for s in df_out.index.unique():
    df_out.loc[[i == s for i in df_out.index], 'gpp_dnn'] = np.asarray(y_pred_sites.get(s))

# Save to a csv, to be processed in R
if len(args.output_file)==0:
    df_out.to_csv(f"../models/preds/dnn_lso_epochs_{args.n_epochs}_patience_{args.patience}.csv")   
else:
    df_out.to_csv("../models/preds/" + args.output_file)