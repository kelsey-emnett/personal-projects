import hyperopt
from hyperopt import hp

hyperopt_model_dict = {
    "rf": {
        'type': 'rf',
        'feature_list': hp.choice("feature_list_rf", ["rf_importance", "rf_rfe"]),
        'max_depth': hp.quniform('max_depth_rf', 2, 14, 1.0),
        'min_samples_split': hp.quniform('min_samples_split_rf', 5, 15, 1.0),
        'n_estimators': hp.quniform('n_estimators_rf', 50, 200, 10.0),
        'max_features': hp.choice("max_features_rf", ['sqrt', 'log2', None])
    },
    "gbt": {
        'type': 'gbt',
        'feature_list': hp.choice("feature_list_gbt", ["gbt_importance", "gbt_rfe"]),
        'max_depth': hp.quniform('max_depth_gbt', 2, 14, 1.0),
        'min_samples_split': hp.quniform('min_samples_split_gbt', 5, 15, 1.0),
        'n_estimators': hp.quniform('n_estimators_gbt', 50, 200, 10.0),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
        'subsample': hp.uniform('subsample_gbt', 0.7, 1.0)
    },
    "elastic_net": {
        'type': 'elastic_net',
        'feature_list': hp.choice("feature_list_elastic_net", ["elastic_net_importance", "elastic_net_rfe"]),
        'l1_ratio': hp.uniform('l1_ratio_elastic_net', 0.2, 0.8),
        'alpha': hp.uniform('alpha_elastic_net', 0.2, 3.0)
    },
    "lasso": {
        'type': 'lasso',
        'feature_list': hp.choice("feature_list_lasso", ["lasso_importance", "lasso_rfe"]),
        'alpha': hp.uniform('alpha_lasso', 0.2, 3.0)
    },
    "ridge": {
        'type': 'ridge',
        'feature_list': hp.choice("feature_list_ridge", ["ridge_importance", "ridge_rfe"]),
        'alpha': hp.uniform('alpha_ridge', 0.2, 3.0),
        'solver': hp.choice("solver", ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'])
    },
    "decision_trees": {
        'type': 'decision_trees',
        'feature_list': hp.choice("feature_list_dt", ["decision_trees_importance", "decision_trees_rfe"]),
        'max_depth': hp.quniform('max_depth_dt', 2, 14, 1.0),
        'criterion': hp.choice("criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"]),
        'min_samples_split': hp.quniform('min_samples_split_dt', 5, 15, 1.0),
        'max_features': hp.choice("max_features_dt", ['sqrt', 'log2', None])
    },
    "knn": {
        'type': 'knn',
        'feature_list': hp.choice("feature_list_knn", ["lasso_importance", "lasso_rfe", 
                                   "ridge_importance", "ridge_rfe", 
                                   "rf_importance", "rf_rfe", 
                                   "gbt_importance", "gbt_rfe"]),
        "n_neighbors": hp.quniform("n_neighbors", 3, 10, 1.0),
        "metric": hp.choice("metric", ["minkowski", "manhattan", "euclidean"])
    },
    "svm": {
        'type': 'svm',
        'feature_list': hp.choice("feature_list_svm", ["lasso_importance", "lasso_rfe", 
                                                       "ridge_importance", "ridge_rfe", 
                                                       "rf_importance", "rf_rfe", 
                                                       "gbt_importance", "gbt_rfe"]),
        'kernel': hp.choice("kernel", ["linear", "poly", "rbf", "sigmoid"]),
        'C': hp.uniform('C', 0.1, 3.0),
        'epsilon': hp.uniform('epsilon', 0.01, 3.0)
    }
}