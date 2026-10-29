# Reinforcement Learning - Temporal Difference

This directory contains implementations of Temporal Difference (TD) learning algorithms applied to OpenAI Gym environments.

## Learning Objectives

- What is Temporal Difference (TD) learning?
- What is Monte Carlo learning?
- What is TD(λ) and how does it bridge TD(0) and Monte Carlo?
- What is SARSA(λ) and how does it differ from Q-learning?
- What are Eligibility Traces and how are they used in backward-view algorithms?

## Files

| File | Description |
| --- | --- |
| `0-monte_carlo.py` | Implements the Monte Carlo algorithm for policy evaluation |
| `1-td_lambtha.py` | Implements the TD(λ) algorithm for value function estimation |
| `2-sarsa_lambtha.py` | Implements the SARSA(λ) algorithm for policy iteration |
| `README.md` | Documentation for the Temporal Difference module |

## Requirements

- Ubuntu 20.04 LTS with Python 3.8
- `pycodestyle` style guidelines (version 2.8.x)
- All files must be executable (`chmod +x`)
- Code must follow POSIX standard (end with a new line)
