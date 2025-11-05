"""
Dataset class for containing single-step dynamics transition data.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class DynamicsDataset(Dataset):
    def __init__(self, data_dict):
        """
        Initializes the DynamicsDataset with a dictionary of transition data.

        Args:
            data_dict (dict): Dictionary containing 'states', 'actions', and 'next_states' keys.
                            Values should be lists of numpy arrays.
        """
        self.data = {}
        for key, value in data_dict.items():
            # Convert list of numpy arrays to a single numpy array, then to tensor
            self.data[key] = torch.from_numpy(np.array(value)).float()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data['states'])

    def __getitem__(self, idx):
        """
        Retrieves the sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing (state, action, next_state).
        """
        return (self.data['states'][idx], self.data['actions'][idx], self.data['next_states'][idx])
