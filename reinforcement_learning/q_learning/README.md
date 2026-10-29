# Reinforcement Learning - Q-Learning

This directory contains implementations of Q-learning algorithms applied to OpenAI Gym environments (specifically `FrozenLake-v1`).

## Learning Objectives

- What is Reinforcement Learning?
- What is an environment, agent, state, action, and reward?
- What is a policy function and value function?
- What is Epsilon-Greedy strategy and how does it balance exploration vs exploitation?
- What is Q-learning and how is the Q-table updated?

## Files

| File | Description |
| --- | --- |
| `0-load_env.py` | Loads the FrozenLake environment |
| `1-q_init.py` | Initializes the Q-table with zeros |
| `2-epsilon_greedy.py` | Implements the epsilon-greedy action selection strategy |
| `3-q_learning.py` | Trains an agent using Q-learning |
| `4-play.py` | Evaluates a trained agent playing an episode |
| `README.md` | Documentation for the Q-Learning module |

## Requirements

- Ubuntu 20.04 LTS with Python 3.8
- `pycodestyle` style guidelines (version 2.8.x)
- All files must be executable (`chmod +x`)
- Code must follow POSIX standard (end with a new line)
