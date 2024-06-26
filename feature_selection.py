from typing import Any
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np

def recursive_feature_selector(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_data_cuts: int = 3, 
    min_features_to_select: int = 1, 
    num_features_to_remove: int = 1, 
    scoring_metric: str = "neg_mean_absolute_error",
) -> list[str]:
    """
    Uses recursive feature selection and model feature importances to remove
    the least informative features from the modeling feature list one-by-one.

    Args:
        model (Any): Scikit-learn prediction model
        X_train (pd.DataFrame): Dataframe with training modeling features
        y_train (pd.DataFrame): Dataframe with training target values
        cv_data_cuts (int): Number of cross-validation data cuts to use for
            feature selection
        min_features_to_select (int): Minimum number of features to select
        num_features_to_remove (int): Number of features to remove at each modeling step
        scoring_metric (str): Scoring metric to use to evaluate feature importance. The
            scoring metrics available for evaluation can be found at this link:
            https://scikit-learn.org/stable/modules/model_evaluation.html
    
    Returns:
        selected_features (list[str]): List with ideal selected modeling features for 
            the provided model
    """

    rfecv = RFECV(
        estimator=model,
        step=num_features_to_remove,
        cv=cv_data_cuts,
        scoring=scoring_metric,
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
    )

    rfecv.fit(X_train, y_train)
    
    num_features = rfecv.n_features_

    rfe = RFE(model, n_features_to_select=num_features, step=num_features_to_remove)
    feature_selector = rfe.fit(X_train, y_train)

    selected_features = list(X_train.columns[feature_selector.support_])

    return selected_features


def feature_importance_selector(
    model: Any, 
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame
) -> list[str]:
    """
    Uses model importance to select the most informative features from the modeling

    Args:
        model (Any): Scikit-learn prediction model
        X_train (pd.DataFrame): Dataframe with training modeling features
        y_train (pd.DataFrame): Dataframe with training target values
    
    Returns:
        selected_features (list[str]): List with ideal selected modeling features for 
            the provided model
    """

    feature_selector = SelectFromModel(model).fit(X_train, y_train)

    selected_features = list(feature_selector.get_feature_names_out())

    return selected_features
