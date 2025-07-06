# Q-Learning CartPole Demo

This notebook demonstrates a basic implementation of the Q-learning reinforcement learning algorithm applied to the classic CartPole-v1 environment from OpenAI Gym.

## Overview

Q-learning is a value-based off-policy reinforcement learning algorithm. It estimates the expected future rewards of state-action pairs and updates these estimates iteratively using the Bellman equation.

In this demo:

- The **CartPole-v1** environment is used, where the agent must learn to balance a pole on a cart.
- The state space is discretized to make it suitable for Q-table based learning.
- The agent uses an epsilon-greedy policy to balance exploration and exploitation.
- The Q-table is updated using the standard Q-learning update rule.

---

## Requirements

This project uses **[PDM](https://pdm.fming.dev/)** as the Python environment and dependency manager. All dependencies are listed in the `pyproject.toml`.

### 🛠️ Setting up with PDM

If you don't have `pdm` installed, you can install it with:

```bash
pip install pdm
```

Then, to set up the environment and install dependencies:

```bash
# Navigate to the project directory (containing pyproject.toml)
cd path/to/project

# Install dependencies and create a virtual environment
pdm install
```

To activate the environment (optional, depending on shell):

```bash
pdm venv activate
```

Then run the notebook using:

```bash
pdm run jupyter notebook
```

---

## Core Components

### 1. Environment Setup

```python
import gym
env = gym.make('CartPole-v1')
```

The environment has:

- **Observation space**: 4 continuous variables (cart position, velocity, pole angle, angular velocity)
- **Action space**: 2 discrete actions (move left or right)

### 2. State Discretization

The continuous state space is discretized using `np.linspace` into fixed bins for each dimension:

```python
def discretize(obs):
    # Convert continuous observation into discrete bins
    ...
```

This allows the use of a Q-table, which maps discrete `(state, action)` pairs to Q-values.

### 3. Q-Table Initialization

The Q-table is initialized with zeros:

```python
q_table = np.zeros(obs_space_size + (action_space_size,))
```

### 4. Q-Learning Algorithm

The agent interacts with the environment across multiple episodes:

- Select action using ε-greedy strategy.
- Update Q-values using the Bellman equation:

  ```python
  q_table[state][action] += learning_rate * (
      reward + discount_factor * np.max(q_table[new_state]) - q_table[state][action]
  )
  ```

- Decay ε over time to reduce exploration.

### 5. Training Loop

The training runs for a configurable number of episodes (`n_episodes`), and tracks average rewards to evaluate learning performance.

---

## Results

- A plot is shown of average reward over time.
- The agent typically learns to balance the pole successfully within a few hundred episodes.
- A final test run demonstrates the trained agent's performance.

---

## Hyperparameters

Key hyperparameters include:

```python
n_episodes = 10000
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.01
```

Adjust these for different learning behavior.

---

## Visualization

- Reward per episode is plotted to visualize learning.
- A final animation (optional, with `render_mode='human'`) can demonstrate performance.

---

## Notes

- This is a simple implementation for educational purposes.
- Deep reinforcement learning (e.g., DQN) is needed for environments with large or continuous state spaces.
- Consider saving and loading Q-tables for long-term use.

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.
- OpenAI Gym Documentation: https://www.gymlibrary.dev/
- PDM: https://pdm.fming.dev/