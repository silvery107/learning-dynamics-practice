"""
Script for training the dynamics model using the collected dataset.
"""

import torch
from torch import optim
from src.dataset.dynamics_dataset import DynamicsDataset
from src.model.dynamics_model import DirectDynamicsModel, ResidualDynamicsModel

def train_model(model, dataset, num_epochs=100, batch_size=64, learning_rate=1e-3):
    """
    Trains the given dynamics model using the provided dataset.

    Args:
        model (torch.nn.Module): The dynamics model to train.
        dataset (DynamicsDataset): The dataset containing training samples.
        num_epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
        learning_rate (float): Learning rate for the optimizer.
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for states, actions, next_states in dataloader:
            optimizer.zero_grad()
            predicted_next_states = model(states, actions)
            loss = loss_fn(predicted_next_states, next_states)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

def eval_model(model, dataset):
    """
    Evaluates the given dynamics model using the provided dataset.

    Args:
        model (torch.nn.Module): The dynamics model to evaluate.
        dataset (DynamicsDataset): The dataset containing evaluation samples.

    Returns:
        float: The average loss over the evaluation dataset.
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    loss_fn = torch.nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for states, actions, next_states in dataloader:
            predicted_next_states = model(states, actions)
            loss = loss_fn(predicted_next_states, next_states)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

if __name__ == "__main__":
    # Load dataset
    dataset = torch.load("dynamics_dataset.pt", weights_only=False)
    print(f"Loaded dataset with {len(dataset)} samples.")

    # Split dataset into training and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    # Initialize model
    direct_model = DirectDynamicsModel(2, 1)

    # Train model
    train_model(direct_model, train_dataset, num_epochs=50, batch_size=32, learning_rate=1e-3)

    # Evaluate model
    val_loss = eval_model(direct_model, val_dataset)
    print(f"Validation Loss: {val_loss:.6f}")

    # Save trained model
    torch.save(direct_model.state_dict(), "direct_dynamics_model.pt")
    print("Trained model saved to direct_dynamics_model.pt")

    # Do the same for the residual model
    residual_model = ResidualDynamicsModel(2, 1)
    train_model(residual_model, train_dataset, num_epochs=50, batch_size=32, learning_rate=1e-3)
    val_loss = eval_model(residual_model, val_dataset)
    print(f"Validation Loss: {val_loss:.6f}")
    torch.save(residual_model.state_dict(), "residual_dynamics_model.pt")
    print("Trained model saved to residual_dynamics_model.pt")
