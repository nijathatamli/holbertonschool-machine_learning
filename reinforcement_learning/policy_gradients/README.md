# Reinforcement Learning - Policy Gradients

This directory contains implementations of Policy Gradient methods applied to OpenAI Gym environments (e.g., CartPole-v0).

## Learning Objectives

- What is a Policy Gradient method and how does it differ from Value-based methods?
- What is the Monte Carlo Policy Gradient (REINFORCE) algorithm?
- How to compute action probabilities using softmax and policy weight matrices
- How to update policy weights using gradient ascent based on episodic returns

## Files

| File | Description |
| --- | --- |
| `policy_gradient.py` | Contains functions to compute the policy distribution and calculate action gradients |
| `train.py` | Implements the training loop for policy gradient optimization |
| `README.md` | Documentation for the Policy Gradients module |

## Requirements

- Ubuntu 20.04 LTS with Python 3.8
- `pycodestyle` style guidelines (version 2.8.x)
- All files must be executable (`chmod +x`)
- Code must follow POSIX standard (end with a new line)
