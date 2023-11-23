# This file contains the training and testing
# functions used in the main scripts

# Import dependencies
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score

def train_loop(dataloader, model, optimizer, DEVICE):

    # Initiate training losses, to aggregate over sites
    train_loss = 0.0
    train_r2 = 0.0

    # Set model to training mode (activates dropout, batch normalization, etc, if present)
    model.train()

    # Loop over all training sites
    for x, y, mask in dataloader:
        # Send tensors to the correct device
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        # Perform forward pass for predictions
        y_pred = model(x)

        # Reset gradient to zero, rather than aggregating gradient over sites
        optimizer.zero_grad()

        # Compute MSE losss between GPP and predicted GPP
        loss = F.mse_loss(y_pred.flatten(), y.flatten())

        # Backpropagate the gradients through the model
        loss.backward()

        # Update model parameters using the optimizer
        optimizer.step()

        # Accumulate the training loss over sites
        train_loss += loss.item()

        # Compute coefficient of determination on training data
        train_r2 += r2_score(y_true = y.detach().cpu().numpy().flatten(),      # format tensor to np.array
                             y_pred = y_pred.detach().cpu().numpy().flatten())

    # Set model to evaluation mode (deactivate droptou, etc)
    model.eval()

    # Return computed training loss
    return train_loss, train_r2



def train_loop_cat(dataloader, model, optimizer, DEVICE):

    # Initiate training losses, to aggregate over sites
    train_loss = 0.0
    train_r2 = 0.0

    # Set model to training mode (activates dropout, batch normalization, etc, if present)
    model.train()

    # Loop over all training sites
    for x, y, c, mask in dataloader:

        # Send tensors to the correct device
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        c = c.to(DEVICE)
        
        # Perform forward pass for predictions
        y_pred = model(x, c)

        # Reset gradient to zero, rather than aggregating gradient over sites
        optimizer.zero_grad()

        # Compute MSE losss between GPP and predicted GPP
        loss = F.mse_loss(y_pred.flatten(), y.flatten())

        # Backpropagate the gradients through the model
        loss.backward()

        # Update model parameters using the optimizer
        optimizer.step()

        # Accumulate the training loss over sites
        train_loss += loss.item()

        # Compute coefficient of determination on training data
        train_r2 += r2_score(y_true = y.detach().cpu().numpy().flatten(),      # format tensor to np.array
                             y_pred = y_pred.detach().cpu().numpy().flatten())

    # Set model to evaluation mode (deactivate droptou, etc)
    model.eval()

    # Return computed training loss
    return train_loss, train_r2


def test_loop(dataloader, model, DEVICE):

    # Set model to evaluation mode
    model.eval()

    # Initiate testing losses, to aggregate over sites (if there are more than one)
    test_loss = 0.0
    test_r2 = 0.0

    # Stop computing gradients during the following code chunk:d
    with torch.no_grad():

        # Get testing data from dataloader
        for x, y, mask in dataloader:

            # Send tensors to the correct device
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # Perform forward pass for predictions
            y_pred = model(x)

            # Squeeze empty tensor dimensions to match y and y_pred
            y = y.squeeze()
            y_pred = y_pred.squeeze()

            # Compute test MSE on testing data (including imputed values)
            test_loss += F.mse_loss(y_pred, y)

            # Transform prediction tensor into numpy array
            y_pred = y_pred.detach().cpu().numpy()

            # Get mask as a Boolean list, from the torch tensor given by the data loader
            mask = mask.numpy()[0]

            # Compute R2 on non-imputed testing data
            test_r2 += r2_score(y_true = y.detach().cpu().numpy()[mask],
                               y_pred = y_pred[mask])
            

    # Return computed testing loss
    return test_loss, test_r2, y_pred


def test_loop_cat(dataloader, model, DEVICE):

    # Set model to evaluation mode
    model.eval()

    # Initiate testing losses, to aggregate over sites (if there are more than one)
    test_loss = 0.0
    test_r2 = 0.0

    # Stop computing gradients during the following code chunk:d
    with torch.no_grad():

        # Get testing data from dataloader
        for x, y, c, mask in dataloader:

            # Send tensors to the correct device
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            c = c.to(DEVICE)

            # Perform forward pass for predictions
            y_pred = model(x, c)

            # Squeeze empty tensor dimensions to match y and y_pred
            y = y.squeeze()
            y_pred = y_pred.squeeze()

            # Compute test MSE on testing data (including imputed values)
            test_loss += F.mse_loss(y_pred, y)

            # Transform prediction tensor into numpy array
            y_pred = y_pred.detach().cpu().numpy()

            # Get mask as a Boolean list, from the torch tensor given by the data loader
            mask = mask.numpy()[0]

            # Compute R2 on non-imputed testing data
            test_r2 += r2_score(y_true = y.detach().cpu().numpy()[mask],
                               y_pred = y_pred[mask])
            

    # Return computed testing loss
    return test_loss, test_r2, y_pred
