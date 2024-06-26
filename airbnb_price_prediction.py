# Databricks notebook source
import pandas as pd
from data_preparation import split_data, preprocess_data, id_highly_correlated_variables, calc_mutual_info_scores, remove_highly_correlated_vars
from feature_selection import recursive_feature_selector, feature_importance_selector
from train_model import fit_model
from hyperopt_utils import Hyperopt

# COMMAND ----------

pandas_df = spark.read.parquet("dbfs:/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet").toPandas()

# COMMAND ----------

# Row count is small. I am going to use sklearn for modeling instead of spark because it will have a larger variety of choices
pandas_df.shape[0]

# COMMAND ----------

display(pandas_df)

# COMMAND ----------

# appears from the above that the data types came in correctly
pandas_df.dtypes

# COMMAND ----------

pandas_df.isnull().sum()

# COMMAND ----------

# It appears that the dataset is pre-imputed and includes a missing flag whereever the missing data has occurred
na_columns = pandas_df.columns[24:35]
pandas_df[na_columns].sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data summary observations
# MAGIC - Price the target variable is very skewed towards higher values, which may make correctly predicting higher prices difficult. However, given numbers shown below Airbnb likely gets most of its revenue from low-to-middle priced Airbnbs, which may mean we want to better predict those values rather than the tails. This means MAE or MAPE may be a better error metric for evaluation than RMSE or MSE, which will focus on predicting outliers well.
# MAGIC - property_type and neighborhood_cleansed have very high cardinality. One-hot-encoding will not be a good method for these features.
# MAGIC - There appear to be no obvious ordinal categorical variables in the category or numeric columns
# MAGIC - Most obvious data cleaning has already been completed by Databricks. Mostly just feature engineering is left.

# COMMAND ----------

dbutils.data.summarize(pandas_df)

# COMMAND ----------

pandas_df.nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC ## High cardinality variables
# MAGIC - For property_type, values are grouped into a small number of values and there is a long tail. I will recode as major types and create an "other" category
# MAGIC - For neighborhood_cleansed, the values are spread out more evently within categories. A leave-one-out target encoding strategy may work better here

# COMMAND ----------

pandas_df["property_type"].value_counts()

# COMMAND ----------

pandas_df["neighbourhood_cleansed"].value_counts()

# COMMAND ----------

# Group smaller property types into a larger "other" category
keep_property_values = list(pandas_df["property_type"].value_counts()[lambda x : x > 80].index)
pandas_df["property_type"] = pandas_df["property_type"].apply(lambda x: x if x in keep_property_values else "Other")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Choice of loss metric
# MAGIC As shown during the EDA, Airbnb's rates are highly skewed. The majority  of Airbnb's revenue likely comes where the bulk of their rates are, low and medium. Only a small number of listings had very high rates. 
# MAGIC
# MAGIC I wanted to predict lower and medium rates better so chose MAE as my loss metric. This metric does not penalize models more harshly for missing higher values. MSE would have been a better choice if predicting higher rates was more important to the business.

# COMMAND ----------

# I'm going to work with the default missing value imputation. 
# It appears an intelligent method was used other than the most simple imputation method.
seed=123
target_col = "price"
prediction_col = "price_predicted"

modeling_df = pandas_df.drop(columns=na_columns)

# COMMAND ----------

pandas_df.head()

# COMMAND ----------

# Split dataframe into training, validation, and test data sets
# Separating target column from features
train, y_train, val, y_val, test, y_test = split_data(modeling_df, target_col)

# COMMAND ----------

print(f"Train data row count is {train.shape[0]}")
print(f"Validation data row count is {val.shape[0]}")
print(f"Holdout/test data row count is {test.shape[0]}")

# COMMAND ----------

# Create lists of features for different data types
high_card_vars = ["neighbourhood_cleansed"]
cat_vars = list(set(train.select_dtypes(include=['object']).columns.to_list()) - set(high_card_vars))
cont_vars = train.select_dtypes(include=['float64']).columns.to_list()

# COMMAND ----------

import mlflow
from mlflow.exceptions import RestException

# Start MLflow experiment
experiment_path = "/Users/contact@kelseyhuntzberry.com/databricks-coding-challenge/mlflow_experiments/"
experiment_name = "airbnb_price_prediction"

try:
    experiment_id = mlflow.create_experiment(f"{experiment_path}{experiment_name}")
    mlflow.set_experiment(experiment_name=f"{experiment_path}{experiment_name}")
except RestException:
    mlflow.set_experiment(experiment_name=f"{experiment_path}{experiment_name}")

# COMMAND ----------

mlflow.sklearn.autolog()

# COMMAND ----------

# Send data through preprocessing pipeline
preprocessor, X_train, X_val, X_test, cat_model_names, num_model_names, hc_model_names = preprocess_data(train, y_train, val, test, cat_vars, cont_vars, high_card_vars)

# COMMAND ----------

mlflow.sklearn.autolog(disable=True)

# COMMAND ----------

# Identify highly correlated variables above the user-defined threshold
cont_var_names = num_model_names + hc_model_names

high_corr_df = id_highly_correlated_variables(X_train, cont_var_names, corr_cutoff=0.7)

high_corr_df

# COMMAND ----------

# Calculate mutual information scores
mi_score_df = calc_mutual_info_scores(X_train, y_train, cont_var_names)

mi_score_df.head()

# COMMAND ----------

# Removing variables with the highest mutual information score that are above the correlation threshold
corr_mi_score_df, remove_variables = remove_highly_correlated_vars(high_corr_df, mi_score_df)

print("remove the following variables:", remove_variables)

corr_mi_score_df.head()

# COMMAND ----------

# Removing highly correlated features
X_train = X_train.drop(columns = remove_variables)
X_val = X_val.drop(columns = remove_variables)
X_test = X_test.drop(columns = remove_variables)

# COMMAND ----------

# Setting up a dictionary to cleanly perform feature selection without a lot of repeated code
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

feature_selection_model_dict = {
    "gbt": GradientBoostingRegressor(random_state=seed, max_depth=5),
    "rf":  RandomForestRegressor(random_state=seed, max_depth=5),
    "ridge":  Ridge(random_state=seed),
    "lasso": Lasso(random_state=seed),
    "elastic_net": ElasticNet(random_state=seed),
    "decision_trees": DecisionTreeRegressor(random_state=seed, max_depth=5),
}

# COMMAND ----------

features_dict = {}

# Performing feature selection for a variety of models
for model_name in list(feature_selection_model_dict.keys()):

    model = feature_selection_model_dict[model_name]

    feature_names = recursive_feature_selector(model, X_train, y_train)

    feature_set_name = f"{model_name}_rfe"

    features_dict[feature_set_name] = feature_names

# COMMAND ----------

# Performing feature selection for a variety of models
for model_name in list(feature_selection_model_dict.keys()):

    model = feature_selection_model_dict[model_name]

    feature_names = feature_importance_selector(model, X_train, y_train)

    feature_set_name = f"{model_name}_importance"

    features_dict[feature_set_name] = feature_names


# COMMAND ----------

import json

with open('./configs/features.json', 'w') as fp:
    json.dump(features_dict, fp)

# COMMAND ----------

mlflow.sklearn.autolog()

# COMMAND ----------

# Run hyperopt hyperparameter tuning across model types
from hyperopt import space_eval

hyperopt_class = Hyperopt(X_train, y_train, X_val, y_val, features_dict, apply_overfit_penalty=True)

best_result, search_space = hyperopt_class.run_hyperopt(max_evals=150)

hyperopt_results = space_eval(search_space, best_result)
print(hyperopt_results)

# COMMAND ----------

# Dynamically extract best hyperparameter values and correct model
final_model_type = hyperopt_results["type"]
final_feature_list_name = hyperopt_results["feature_list"]
final_feature_list = features_dict[final_feature_list_name]

final_model_params = hyperopt_results.copy()
del final_model_params["type"]
del final_model_params["feature_list"]

final_model_params = hyperopt_class.round_hyperparameters(final_model_params)

final_model = hyperopt_class.extract_model(model_name=final_model_type, **final_model_params)

# COMMAND ----------

# Do full evaluation on model with validation data
final_model, final_val_results, final_feature_importances = fit_model(
    final_model, 
    features_dict[final_feature_list_name], 
    X_train, 
    y_train, 
    X_val, 
    y_val, 
    final_model_type, 
    final_feature_list_name
)

# COMMAND ----------

final_val_results

# COMMAND ----------

final_feature_importances.head()

# COMMAND ----------

# Evaluation model with holdout framework
holdout_model, holdout_results, holdout_feature_importances = fit_model(
    final_model, 
    features_dict[final_feature_list_name], 
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    final_model_type, 
    final_feature_list_name
)

# COMMAND ----------

holdout_results

# COMMAND ----------

holdout_feature_importances.head()

# COMMAND ----------


