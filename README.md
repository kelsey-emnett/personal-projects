# Airbnb Price Prediction

## Overview
This project creates a regression model that predicts Airbnb prices in San Francisco. It uses a Databricks-provided dataset.

## Environment
This code was built in Databricks and uses Databricks 14.3 ML Runtime. Code linting was run with pre-commit and ruff. Python 3 was used. Only single-node scikit-learn models were used. Therefore, the workflow can be run on a single-node cluster.

## Project Goals
This project aims to follow best practices for a regression model like removing highly correlated variables and parallelized hyperparameter tuning. It aims to achieve this in a reproducible way so that it can easily modified and used for new use cases.

## Project Features:
- Strategically and automatically removes highly correlated variables
- Performs multiple types of model-specific feature selection techniques
- Runs parallelized hyperparameter tuning with Hyperopt using Bayesian optimization for optimal performance and model accuracy
- Uses MLflow for model and experiment tracking
