# This module defines two LSTM-based neural networks for GPP prediction (regression). 
# The first model takes input numerical sequences (x) to predict an output sequence (y). 
# The ModelCond network architecture is conditioned on categorical features (c)
# in addition.

import torch.nn as nn
import torch


class ModelCond(nn.Module):
    def __init__(self, input_dim, conditional_dim, hidden_dim, num_layers=2):
        super().__init__()

        # LSTM layer for sequence processing
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, dropout=0.3)

        # Categorical features are used (vegetation class and land use), concatenated to the hidden state
        self.fc1 = nn.Sequential(
        nn.Linear(in_features=hidden_dim+conditional_dim, out_features=64),
        nn.ReLU()
        )
        # Fully connected layers for feature processing
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU()
        )
        self.fc3= nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU()
        )
        
        # Final linear layer for regression output
        self.fc4 = nn.Linear(16, 1)
        
    def forward(self, x, c):
        # Forward pass through the LSTM layer
        out, (h,d) = self.lstm(x)

        # Concatenate conditional features
        out = torch.cat([out,c], dim=2)
        
        # Pass the concatenated output through fully connected layers
        y = self.fc1(out)
        y = self.fc2(y)
        y = self.fc3(y)
        y = self.fc4(y)

        return y



class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim,  num_layers=2):
        super().__init__()

        # LSTM layer for sequence processing
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, dropout=0.3)

        # Fully connected layers for feature processing
        self.fc1 = nn.Sequential(
        nn.Linear(in_features=hidden_dim, out_features=64),
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
        
        # Final linear layer for regression output
        self.fc4 = nn.Linear(16, 1)
        
    def forward(self, x):
        # Forward pass through the LSTM layer
        out, (h,d) = self.lstm(x)
        
        # Pass the concatenated output through fully connected layers
        y = self.fc1(out)
        y = self.fc2(y)
        y = self.fc3(y)
        y = self.fc4(y)

        return y
