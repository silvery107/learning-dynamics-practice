"""
Dynamics model class.
"""

import torch
from torch import nn

class DirectDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Initializes the DynamicsModel with given dimensions.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(DirectDynamicsModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        """
        Forward pass through the model.

        Args:
            state (torch.Tensor): Current state tensor.
            action (torch.Tensor): Action tensor.

        Returns:
            torch.Tensor: Predicted next state tensor.
        """
        x = torch.cat([state, action], dim=-1)
        next_state = self.model(x)
        return next_state
    

class ResidualDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Initializes the ResidualDynamicsModel with given dimensions.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(ResidualDynamicsModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        """
        Forward pass through the model.

        Args:
            state (torch.Tensor): Current state tensor.
            action (torch.Tensor): Action tensor.

        Returns:
            torch.Tensor: Predicted next state tensor.
        """
        x = torch.cat([state, action], dim=-1)
        delta_state = self.model(x)
        next_state = state + delta_state
        return next_state