from typing import Tuple, Any
from category_encoders.leave_one_out import LeaveOneOutEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import mlflow
import pandas as pd
import numpy as np


def split_data(
    df: pd.DataFrame, 
    target_col: str, 
    validation_proportion: float = 0.2, 
    test_proportion: float = 0.1, 
    seed: int = 123,
) -> Tuple[
    pd.DataFrame, 
    np.ndarray, 
    pd.DataFrame, 
    np.ndarray
]:
    """
    Randomly splits data into training, validation, and holdout data sets
    for features and target variables based on user-defined proportions.

    Args:
        df (pd.DataFrame): Cleansed dataframe with target variable and feature columns
        target_col (str): String with target column name
        validation_proportion (float): Proportion of data to devote to the validation
            data set
        test_proportion (float): Proportion of data to devoote to the holdout data set
        seed (int): Random number setting the seed to keep results consistent
    
    Returns:
        train (pd.DataFrame): Dataframe with feature columns with randomly chosen
            training rows
        y_train (np.ndarray): Array with target column with randomly chosen training rows
        val (np.ndarray): Dataframe with feature columns with randomly chosen validation rows
        y_val (np.ndarray): Array with target column with randomly chosen validation rows
        test (pd.DataFrame): Dataframe with feature columns with randomly chosen holdout rows
        y_test: (np.ndarray): Array with target column with randomly chosen holdout rows
    """
    features_df = df.drop(columns=target_col)
    target_df = df[target_col]

    first_split_proportion = validation_proportion + test_proportion
    second_split_proportion = test_proportion / first_split_proportion

    train, val_temp, y_train, y_val_temp = train_test_split(
        features_df, 
        target_df,
        test_size=first_split_proportion, 
        random_state=seed
    )
    val, test, y_val, y_test = train_test_split(
        val_temp, 
        y_val_temp, 
        test_size=second_split_proportion, 
        random_state=seed
    )

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    return train, y_train, val, y_val, test, y_test


def preprocess_data(
    train: pd.DataFrame, 
    y_train: np.ndarray, 
    val: pd.DataFrame, 
    test: pd.DataFrame, 
    cat_vars: list[str], 
    cont_vars: list[str], 
    high_card_vars: list[str],
) -> Tuple[
    Any,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
    list[str],
]:
    """
    Applies data pipeline to preprocess (rescale, one-hot encode, etc.) features dataframe.

    Args:
        train (pd.DataFrame): Training dataframe with feature columns
        y_train (np.ndarray): Array with training target values
        val (pd.DataFrame): Validation dataframe with feature columns
        test (pd.DataFrame): Holdout dataframe with feature columns
        cat_vars (list[str]): List with categorical column names prior to preprocessing
        cont_vars (list[str]): List with continuous column names prior to preprocessing
        high_card_vars (list[str]): List with high-cardinality column names prior to preprocessing
    
    Returns:
        preprocessor (Any): Sklearn preprocessing pipeline
        X_train (pd.DataFrame): Preprocessed training dataset with features
        X_val (pd.DataFrame): Preprocessed validation dataset with features
        X_test (pd.DataFrame): Preprocessed test dataset with features
        cat_feature_names (list[str]): List with categorical feature names
        num_feature_names (list[str]): List with numerical feature names
        hc_feature_names (list[str]): List with high-cardinality feature names
    """
    with mlflow.start_run(run_name="preprocessor"):

        scaler = StandardScaler()
        ohe = OneHotEncoder(sparse=False)
        loo_encoder = LeaveOneOutEncoder()

        cat_transformer = Pipeline(
            steps = [("ohe", ohe)]
        )

        num_transformer = Pipeline(
            steps = [("scaler", scaler)]
        )

        hc_transformer = Pipeline(
            steps = [("loo", loo_encoder), ("scaler", scaler)]
        )

        preprocess_transformer = ColumnTransformer(
            transformers = [
                ("cat", cat_transformer, cat_vars),
                ("num", num_transformer, cont_vars),
                ("hc", hc_transformer, high_card_vars),
            ]
        )

        preprocessor = preprocess_transformer.fit(train, y_train)

        cat_feature_names = list(preprocessor.transformers_[0][1].get_feature_names_out())
        num_feature_names = preprocessor.transformers_[1][2]
        hc_feature_names = preprocessor.transformers_[2][2]

        feature_list = cat_feature_names + num_feature_names + hc_feature_names

        X_train = pd.DataFrame(preprocessor.transform(train), columns=feature_list)
        X_val = pd.DataFrame(preprocessor.transform(val), columns=feature_list)
        X_test = pd.DataFrame(preprocessor.transform(test), columns=feature_list)

    return preprocessor, X_train, X_val, X_test, cat_feature_names, num_feature_names, hc_feature_names


def id_highly_correlated_variables(
    X_train: pd.DataFrame, 
    numeric_var_names: list[str], 
    corr_cutoff: float = 0.8,
) -> pd.DataFrame:
    """
    Function to identify highly correlated features so one feature can be removed

    Args:
        X_train (pd.DataFrame): Preprocessed training dataset with features
        numeric_var_names (list[str]): List with numerical feature names
        corr_cutoff (float): Cutoff above which one column in a pair of correlated
            features will be removed
    
    Returns:
        high_corr_df (pd.DataFrame): Dataframe with feature pairs with correlation
            above the provided threshold value
    """

    corr_matrix = X_train[numeric_var_names].corr().reset_index()

    corr_df = corr_matrix.melt(id_vars='index')

    corr_df = corr_df[corr_df["index"] != corr_df["variable"]]

    corr_df.rename(columns={"index":"variable1",
                            "variable":"variable2",
                            "value":"corr"},inplace=True)
    
    corr_df = corr_df.sort_values("corr")

    corr_df["variable1_lag"] = corr_df["variable1"].shift(1)
    corr_df = corr_df[corr_df["variable2"] != corr_df["variable1_lag"]]
    corr_df.drop(columns="variable1_lag", inplace=True)

    high_corr_df = corr_df[abs(corr_df["corr"]) >= corr_cutoff].reset_index(drop=True)
    
    return high_corr_df


def calc_mutual_info_scores(
    X_train: pd.DataFrame, 
    y_train: np.ndarray, 
    modeling_var_names: list[str],
) -> pd.DataFrame:
    """
    Calculates mutual information scores for numeric variables so that the 
    most informative feature of the highly correlated pair can be kept.

    Args:
        X_train (pd.DataFrame): Preprocessed training dataset with features
        y_train (np.ndarray): Array with target variable actual values
        modeling_var_names (list[str]): List with modeling feature names
    
    Returns:
        mi_score_df (pd.DataFrame): Dataframe with mutual information analysis
            results for feature pairs
    """

    mi_score = mutual_info_regression(X_train[modeling_var_names], y_train)

    mi_score_df = pd.DataFrame(list(mi_score), columns=["mi_score"])
    mi_score_df["features"] = modeling_var_names

    mi_score_df = mi_score_df[["features", "mi_score"]]

    return mi_score_df


def remove_highly_correlated_vars(
    high_corr_df: pd.DataFrame, 
    mi_score_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    list[str],
]:
    """
    Uses mutual information scores to remove least informative feature if correlation
    is above the user-provided correlation coefficient threshold.

    Args:
        high_corr_df (pd.DataFrame): Dataframe with feature pairs with correlation
            above the provided threshold value
        mi_score_df (pd.DataFrame): Dataframe with mutual information analysis
            results for feature pairs
    
    Returns:
        corr_mi_score_df (pd.DataFrame): Dataframe with feature pairs with correlation 
            coefficients and mutual information score results for highly correlated features
        remove_variables (list[str]): List of highly correlated variables to remove from 
            modeling datasets
    """

    corr_mi_score_df = high_corr_df.merge(mi_score_df, left_on="variable1", right_on="features", how="left")
    corr_mi_score_df.drop(columns="features",inplace=True)
    corr_mi_score_df.rename(columns={"mi_score":"mi_score_variable1"}, inplace=True)

    corr_mi_score_df = corr_mi_score_df.merge(mi_score_df, left_on="variable2", right_on="features", how="left")

    corr_mi_score_df.drop(columns="features", inplace=True)
    corr_mi_score_df.rename(columns={"mi_score":"mi_score_variable2"},inplace=True)

    corr_mi_score_df["variable_to_drop"] = corr_mi_score_df.apply(
        lambda row: row["variable2"] if (row['mi_score_variable1'] > row['mi_score_variable2']) else row['variable1'], axis=1
    )

    remove_variables = list(corr_mi_score_df["variable_to_drop"].unique())

    return corr_mi_score_df, remove_variables
