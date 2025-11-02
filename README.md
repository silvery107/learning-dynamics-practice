# Practice for Dynamics Model Learning


## Dependencies
Check [requirements.txt](requirements.txt), or use Google Colab (coming soon)!

## Basic Usage

### 1. Collect Training Data

Collect state-action-next_state transition data from the pendulum environment:

```bash
python collect_data.py
```

This will:
- Create a pendulum environment
- Collect 1000 random samples
- Save the dataset to `dynamics_dataset.pt`

### 2. Train Dynamics Models

Train both direct and residual dynamics models:

```bash
python train_model.py
```

This will:
- Load the collected dataset
- Split into training (80%) and validation (20%) sets
- Train a **Direct Dynamics Model** (predicts next state directly)
- Train a **Residual Dynamics Model** (predicts the residual between current state and next state)
- Save trained models to `direct_dynamics_model.pt` and `residual_dynamics_model.pt`

### 3. Visualize Results in Jupyter Notebook
Explore visualization of the collected dataset and comparison of performance between models in [visualization.ipynb](visualization.ipynb).
