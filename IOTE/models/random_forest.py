import numpy as np
from src.data_preprocess import load_data, split_features_labels
from src.model_evaluation import search_hyperparameters, model_evaluation
from sklearn.ensemble import RandomForestRegressor


class CustomRandomForest(RandomForestRegressor):
    def __init__(self, n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                 warm_start=False, ccp_alpha=0.0, max_samples=None):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                         bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state,
                         verbose=verbose, warm_start=warm_start, ccp_alpha=ccp_alpha, max_samples=max_samples)


if __name__ == '__main__':
    train_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_train.csv'
    test_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_test.csv'
    df_train = load_data(train_file_path)
    df_test = load_data(test_file_path)
    df_train_x, df_train_y = split_features_labels(df_train)
    df_test_x, df_test_y = split_features_labels(df_test)

    # RandomForest Key Parameters Search Space
    param_grid = {
        'n_estimators': np.arange(50, 300, 50).tolist(),  # From 50 to 300, in steps of 50
        'max_depth': list(np.arange(5, 20, 5))  # Between 5 and 20 in steps of 5
    }

    # Using Pearson correlation coefficient to select features
    feature_engineering_params1 = {
        'strategy': 'pearson',
        'threshold': 0.7
    }

    # Using PCA as a feature engineering strategy
    feature_engineering_params2 = {
        'strategy': 'pca',
        'n_components': 0.95
    }

    # Using polynomial feature generation
    feature_engineering_params3 = {
        'strategy': 'kpca',
        'n_components': 20,
        'gamma': 0.1
    }

    # Performing a hyperparameter search
    best_params, best_metrics = search_hyperparameters(CustomRandomForest, param_grid, df_train_x, df_train_y,
                                                       feature_engineering_params=feature_engineering_params3)

    # Make sure n_estimators and max_depth are integers
    best_params = {key: int(value) if key in ['n_estimators', 'max_depth'] else value for key, value in
                   best_params.items()}

    # Evaluate the best model
    metrics = model_evaluation(CustomRandomForest(**best_params), df_train_x, df_train_y, df_test_x, df_test_y,
                               feature_engineering_params=feature_engineering_params3)
