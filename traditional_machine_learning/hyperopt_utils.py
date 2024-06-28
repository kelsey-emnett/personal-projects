from typing import Any, Tuple
import configs.hyperopt_config as hyperopt_config
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import mlflow
import numpy as np
import pandas as pd

INTEGER_PARAMETERS = ["max_depth", "min_samples_split", "n_estimators", "n_neighbors"]


class Hyperopt:
    """
    Class that performs extensive hyperparameter evaluation with Hyperopt to identify
    the ideal model type and hyperparameters.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        features_dict: dict[str, list[str]],
        apply_overfit_penalty: bool = False,
        overfit_penalty_value: float = 0.1,
        overfit_percent_cutoff: float = 0.1,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.features_dict = features_dict
        self.apply_overfit_penalty = apply_overfit_penalty
        self.overfit_penalty_value = overfit_penalty_value
        self.overfit_percent_cutoff = overfit_percent_cutoff

    def round_hyperparameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Rounds hyperparameters that are required to be integers in the
        Hyperopt package because they are not automatically converted.

        Args:
            params (dict[str, Any]): Dictionary with hyperparameters prior to
                processing

        Returns:
            params (dict[str, Any]): Dictonary with hyperparameters after integer
                hyperparameters have been converted to integers
        """

        for parameter_name in INTEGER_PARAMETERS:
            if parameter_name in params:
                params[parameter_name] = int(params[parameter_name])
            else:
                pass

        return params

    def subset_hyperopt_models(
        self, subset_model_list: list[str]
    ) -> list[dict[str, dict[str, Any]]]:
        """
        Subsets models to hyperparameter tune with a user-provided list

        Args:
            subset_model_list (list[str]): Models on which to run hyperparameter tuning

        Returns:
            hyperopt_search_space (dict[str, dict[str, Any]]): Hyperopt search space for
                user-chosen models
        """

        model_dict = hyperopt_config.hyperopt_model_dict.copy()
        hyperopt_search_space = []

        for model_name in subset_model_list:
            hyperopt_search_space.append(model_dict[model_name])

        return hyperopt_search_space

    def extract_model(
        self,
        model_name: str,
        **params: dict[str, Any],
    ) -> Any:
        """
        Function used to extract a model by type during hyperopt hyperparameter
        tuning and after a model is chosen.

        Args:
            model_name (str): String value specifying model type to extract
            params (dict[str, Any]): Dictionary with model hyperparameters

        Returns:
            model (Any): Scikit-learn model of specified type with given hyperparameters
        """

        if model_name == "gbt":
            model = GradientBoostingRegressor(**params)
        elif model_name == "rf":
            model = RandomForestRegressor(**params)
        elif model_name == "decision_trees":
            model = DecisionTreeRegressor(**params)
        elif model_name == "ridge":
            model = Ridge(**params)
        elif model_name == "elastic_net":
            model = ElasticNet(**params)
        elif model_name == "lasso":
            model = Lasso(**params)
        elif model_name == "svm":
            model = SVR(**params)
        elif model_name == "knn":
            model = KNeighborsRegressor(**params)

        return model

    def objective(
        self,
        params: dict[str, Any],
    ) -> dict[str, float | str]:
        """
        Objective function used for hyperopt training. Evaluates a model with
        given hyperparameters. Provides a penalty for overfitting if directed
        by the user.

        Args:
            params (dict[str, Any]): Dictionary with model hyperparameters

        Returns:
            dict[str, Any]: Dictionary with modeling evaluation metric and final
                modeling status
        """

        X_train = self.X_train
        X_val = self.X_val
        y_train = self.y_train
        y_val = self.y_val

        regressor_type = params["type"]
        feature_list = self.features_dict[params["feature_list"]]
        del params["type"]
        del params["feature_list"]

        params = self.round_hyperparameters(params)
        model = self.extract_model(regressor_type)

        fitted_model = model.fit(X_train[feature_list], y_train)
        train_predictions = fitted_model.predict(X_train[feature_list])
        val_predictions = fitted_model.predict(X_val[feature_list])

        val_mae = mean_absolute_error(y_val, val_predictions)
        train_mae = mean_absolute_error(y_train, train_predictions)

        if self.apply_overfit_penalty:
            train_val_pct_diff = (val_mae - train_mae) / train_mae
            if (train_mae > val_mae) and (
                train_val_pct_diff < -(self.overfit_percent_cutoff)
            ):
                final_metric = val_mae * (1 + self.overfit_penalty_value)
            else:
                final_metric = val_mae
        else:
            final_metric = val_mae

        return {"loss": final_metric, "status": STATUS_OK}

    def run_hyperopt(
        self,
        subset_model_list: list[str] = [
            "rf",
            "gbt",
            "decision_trees",
            "ridge",
            "lasso",
            "elastic_net",
            "knn",
            "svm",
        ],
        max_evals: int = 50,
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Runs hyperopt hyperparameter tuning with provided search space and models.
        Performs Bayesian optimization to efficiently search the feature space and
        find the best model.

        Args:
            subset_model_list (list[str]): List with list of models to include in
                hyperparameters tuning
            max_evals (int): Number of models to train to identify the ideal hyperparameters

        Returns:
            best_result (dict[str, Any]): Dictionary with the best hyperparameters used to
                create the chosen model
            search_space (dict[str, dict[str, Any]]): Search space used to run hypopt. This is
                used in combination with the best_result dictionary to obtain the final
                hyperparameters
        """
        algo = tpe.suggest
        spark_trials = SparkTrials()
        model_list = self.subset_hyperopt_models(subset_model_list)
        search_space = hp.choice("regressor_type", model_list)

        with mlflow.start_run():
            best_result = fmin(
                fn=self.objective,
                space=search_space,
                algo=algo,
                max_evals=max_evals,
                trials=spark_trials,
            )

        return best_result, search_space
