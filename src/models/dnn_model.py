# This module defines a PyTorch model for GPP predictions (regression task). 
# It consists of four fully connected layers with ReLU activation functions.
# The architecture matches the fully connected layers, with same dimensions,
# that are defined on top of the LSTM cells in lstm_model.py

import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # First fully connected layer with input features to 64 hidden units
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=64),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU()
        )
        self.fc3= nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU()
        )
        
        # Final linear layer for regression output with 16 to 1 output units
        self.fc4 = nn.Linear(16, 1)
        
    def forward(self, x):
        # Define the forward pass through the neural network

        # Pass input tensor 'x' through the defined fully connected layers
        y = self.fc1(x)
        y = self.fc2(y)
        y = self.fc3(y)
        y = self.fc4(y)
        
        return y