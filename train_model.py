from typing import Any, Tuple
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np


def fit_model(
    model: Any,
    feature_list: list[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    model_type: str,
    feature_list_name: list[str],
) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
    """
    Trains and evaluates a model and providing model feature importances.

    Args:
        model (Any): Scikit-learn model to train
        feature_list (list[str]): List with feature column names
        X_train (pd.DataFrame): Training data set with model features
        y_train (np.ndarray): Array with training data actual values
        X_val (pd.DataFrame): Validation data set with model features
        y_val (np.ndarray): Array with validation data actual values
        model_type (str): String with model type
        feature_list_name (list[str]): String with name of the model feature list

    Returns:
        fitted_model (Any): Scikit-learn model fitted with training dataset
        model_results (pd.DataFrame): Dataframe with model evaluation results
        feature_importances (pd.DataFrame): Dataframe with model feature importance results
    """
    fitted_model = model.fit(X_train[feature_list], y_train)

    predictions_train = fitted_model.predict(X_train[feature_list])
    predictions = fitted_model.predict(X_val[feature_list])

    mse = mean_squared_error(y_val, predictions)
    mape = mean_absolute_percentage_error(y_val, predictions)
    mae = mean_absolute_error(y_val, predictions)

    mse_train = mean_squared_error(y_train, predictions_train)
    mape_train = mean_absolute_percentage_error(y_train, predictions_train)
    mae_train = mean_absolute_error(y_train, predictions_train)

    model_results = pd.DataFrame(
        [
            [
                model_type,
                feature_list_name,
                mse,
                mape,
                mae,
                mse_train,
                mape_train,
                mae_train,
            ]
        ],
        columns=[
            "model",
            "features",
            "mse_val",
            "mape_val",
            "mae_val",
            "mse_train",
            "mape_train",
            "mae_train",
        ],
    )

    if hasattr(fitted_model, "feature_importances_"):
        feature_importances = pd.DataFrame(
            pd.Series(feature_list), columns=["features"]
        )
        feature_importances["importance"] = fitted_model.feature_importances_
        feature_importances = feature_importances.sort_values(
            "importance", ascending=False
        )
    elif hasattr(fitted_model, "coef_"):
        feature_importances = pd.DataFrame(
            pd.Series(feature_list), columns=["features"]
        )
        feature_importances["importance"] = fitted_model.coef_
        feature_importances = feature_importances.sort_values(
            "importance", ascending=False
        )
    else:
        feature_importances = pd.DataFrame([])

    return fitted_model, model_results, feature_importances
