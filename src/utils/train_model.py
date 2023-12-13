# This file contains the main training function

# Dependencies
import numpy as np

# Import functions
from data.dataloader import *
from utils.train_test_loops import *
from utils.train_test_split import train_test_split_sites
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def train_model(data, model, optimizer, writer, n_epochs, DEVICE, patience):
    """
    Trains a PyTorch model, using a train-validation split, with training on all training sites per epoch
    and early stopping based on the mean squared error (MSE) for validation sites.

    Args:
    - data (DataFrame): Pandas DataFrame containing the training data.
    - model (torch.nn.Module): PyTorch model to be trained.
    - optimizer (torch.optim.Optimizer): Optimizer used for training the model.
    - writer (SummaryWriter): TensorBoard SummaryWriter for logging.
    - n_epochs (int): Maximum number of training epochs.
    - DEVICE (str): Device to run the model on (e.g., 'cuda:0' or 'cpu').
    - patience (int): An integer used as patience for early stopping.

    Returns:
    - best_r2 (float): R2 score obtained during training, associated to the epoch with lowest validation MSE.
    - train_mean (array): Mean values used for normalizing the training data, to be reused in model testing.
    - train_std (array): Standard deviation values used for normalizing the training data, to be reused in model testing.
    """

    # Separate train-val split
    data_train, data_val, sites_train, sites_val = train_test_split_sites(data)

    # Calculate mean and standard deviation to normalize the data
    train_mean, train_std = compute_center(data_train)

    # Format pytorch dataset for the data loader
    # Normalize training and validation data according to the training center
    train_ds = gpp_dataset(data_train, train_mean, train_std)
    val_ds = gpp_dataset(data_val, train_mean, train_std)

    # Run data loader with batch_size = 1
    # Due to different time series lengths per site,
    # we cannot load several sites per batch
    train_dl = DataLoader(train_ds, batch_size = 1, shuffle = True)
    val_dl = DataLoader(val_ds, batch_size = 1, shuffle = True)

    
    # Start recording loss (MSE) after each epoch, initialise at Inf
    best_loss = np.Inf

    # Initialize best model
    best_model = None

    # Train the model
    for epoch in range(n_epochs):
        
        # Perform one round of training, doing backpropagation for each training site
        # Obtain the cumulative MSE (training loss) and R2
        train_loss, train_r2 = train_loop(train_dl, model, optimizer, DEVICE)

        # Log tensorboard training values
        writer.add_scalar("mse_loss/train", train_loss, epoch)
        writer.add_scalar("r2_mean/train", train_r2, epoch)         # summed R2, will not be in [0,1] 

        # Evaluate model on test set, removing imputed GPP values
        val_loss, val_r2, y_pred = test_loop(val_dl, model, DEVICE)
        
        # Log tensorboard testing values
        writer.add_scalar("mse_loss/validation", val_loss, epoch)
        writer.add_scalar("r2_mean/validation", val_r2, epoch)

        # Save the model from the best epoch, based on the validation loss
        if val_loss <= best_loss:

            best_loss = val_loss
            # Save the best model's state dictionary
            best_model = model.state_dict()
            # Save the best model's R2 score
            best_r2 = val_r2

            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Check for early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. No improvement in validation loss for {patience} epochs.")
            break

    # Load the best model's weights and return the model object
    if best_model:
        model.load_state_dict(best_model)

    return best_r2, train_mean, train_std



def train_model_cat(data, data_cat, model, optimizer, writer, n_epochs, DEVICE, patience):
    """
    Trains a PyTorch model with two input datasets (numerical and categorical), using a train-validation split, 
    with training on all training sites per epoch
    and early stopping based on the mean squared error (MSE) for validation sites.

    Args:
    - data (DataFrame): Pandas DataFrame containing the numerical training data.
    - data_cat (DataFrame): Pandas DataFrame containing the categorical training data as dummy variables.
    - model (torch.nn.Module): PyTorch model to be trained.
    - optimizer (torch.optim.Optimizer): Optimizer used for training the model.
    - writer (SummaryWriter): TensorBoard SummaryWriter for logging.
    - n_epochs (int): Maximum number of training epochs.
    - DEVICE (str): Device to run the model on (e.g., 'cuda:0' or 'cpu').
    - patience (int): An integer used as patience for early stopping.

    Returns:
    - best_r2 (float): R2 score obtained during training, associated to the epoch with lowest validation MSE.
    - train_mean (array): Mean values used for normalizing the training data, to be reused in model testing.
    - train_std (array): Standard deviation values used for normalizing the training data, to be reused in model testing.
    """

    # Separate train-val split
    data_train, data_val, sites_train, sites_val = train_test_split_sites(data)

    # Separate categorical variables into train-val
    data_cat_train = data_cat.loc[[any(site == s for s in sites_train) for site in data_cat.index]]
    data_cat_val = data_cat.loc[[any(site == s for s in sites_val) for site in data_cat.index]]

    # Calculate mean and standard deviation to center the data
    train_mean, train_std = compute_center(data_train)

    # Format pytorch dataset for the data loader
    train_ds = gpp_dataset_cat(data_train, data_cat_train, train_mean, train_std)
    val_ds = gpp_dataset_cat(data_val, data_cat_val, train_mean, train_std)

    # Run data loader with batch_size = 1
    # Due to different time series lengths per site,
    # we cannot load several sites per batch
    train_dl = DataLoader(train_ds, batch_size = 1, shuffle = True)
    val_dl = DataLoader(val_ds, batch_size = 1, shuffle = True)

    
    # Start recording loss (MSE) after each epoch, initialise at Inf
    best_loss = np.Inf

    # Initialize best model
    best_model = None

    # Train the model
    for epoch in range(n_epochs):
        
        # Perform one round of training, doing backpropagation for each training site
        # Obtain the cumulative MSE (training loss) and R2
        train_loss, train_r2 = train_loop_cat(train_dl, model, optimizer, DEVICE)

        # Log tensorboard training values
        writer.add_scalar("mse_loss/train", train_loss, epoch)
        writer.add_scalar("r2_mean/train", train_r2, epoch)         # summed R2, will not be in [0,1] 

        # Evaluate model on test set, removing imputed GPP values
        val_loss, val_r2, y_pred = test_loop_cat(val_dl, model, DEVICE)
        
        # Log tensorboard testing values
        writer.add_scalar("mse_loss/validation", val_loss, epoch)
        writer.add_scalar("r2_mean/validation", val_r2, epoch)

        # Save the model from the best epoch, based on the validation loss
        if val_loss <= best_loss:

            best_loss = val_loss
            # Save the best model's state dictionary
            best_model = model.state_dict()
            # Save the best model's R2 score
            best_r2 = val_r2

            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Check for early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. No improvement in validation loss for {patience} epochs.")
            break

    # Load the best model's weights and return the model object
    if best_model:
        model.load_state_dict(best_model)

    return best_r2, train_mean, train_std
