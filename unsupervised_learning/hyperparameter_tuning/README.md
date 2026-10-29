# Hyperparameter Tuning

This project covers techniques for optimizing hyperparameters in machine learning models, focusing on **Gaussian Processes** and **Bayesian Optimization**.

## Learning Objectives

- What is a Hyperparameter Tuning?
- What is a Gaussian Process?
- What is a Covariance function / Kernel?
- What is Bayesian Optimization?
- What is an Acquisition Function?
- How to use `GPy` and `GPyOpt` libraries.

## Tasks

| File | Description |
| --- | --- |
| `0-gp.py` | Initializes a noiseless 1D Gaussian Process |
| `1-gp.py` | Calculates the covariance matrix between points in a Gaussian Process |
| `2-gp.py` | Predicts the mean and variance of points in a Gaussian Process |
| `3-bayes_opt.py` | Initializes a Bayesian Optimization on a 1D Gaussian Process |
| `4-bayes_opt.py` | Determines the next point to sample using Bayesian Optimization |
| `5-bayes_opt.py` | Optimizes a 1D Gaussian Process using Bayesian Optimization |
| `6-gpyopt.py` | Optimizes a machine learning model using GPyOpt |

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All files are executed on Ubuntu 20.04 LTS using `python3` (version 3.8)
- Code must adhere to `pycodestyle` style guidelines (version 2.8.x)
- All files must be executable (`chmod +x`)
- All files must end with a new line
