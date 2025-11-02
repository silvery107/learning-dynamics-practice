"""
Dataset class for containing single-step dynamics transition data.
"""
import torch
from torch.utils.data import Dataset

class DynamicsDataset(Dataset):
    def __init__(self, states, actions, next_states):
        """
        Initializes the DynamicsDataset with observations, actions, and next observations.

        Args:
            observations (list or np.array): The current state observations.
            actions (list or np.array): The actions taken.
            next_observations (list or np.array): The resulting state observations after taking the actions.
        """
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
        self.next_states = torch.tensor(next_states, dtype=torch.float32)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.states)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing (observation, action, next_observation).
        """
        return (self.states[idx], self.actions[idx], self.next_states[idx])
