"""
Script for collecting data from the environment and storing it in a dataset.
"""
import torch
from src.env.pendulum_env import PendulumEnv
from src.dataset.dynamics_dataset import DynamicsDataset

def collect_data(env, num_samples, sample_action=False):
    data = {
        'states': [],
        'actions': [],
        'next_states': []
    }

    state = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample() * sample_action
        next_state, _, done, _ = env.step(action)

        data['states'].append(state)
        data['actions'].append(action)
        data['next_states'].append(next_state)

        state = next_state
        if done:
            state = env.reset()

    dataset = DynamicsDataset(data)
    return dataset

if __name__ == "__main__":
    # Create environment
    env = PendulumEnv()
    # Collect data
    print(f"Collecting data...")
    dataset = collect_data(env, num_samples=1000)
    print(f"Collected {len(dataset)} samples.")
    # Save dataset to disk
    filename = "dynamics_dataset.pt"
    torch.save(dataset, filename)
    print(f"Dataset saved to {filename}.")
