import numpy as np
from src.data_preprocess import load_data, split_features_labels
from src.model_evaluation import search_hyperparameters, model_evaluation
from sklearn.linear_model import BayesianRidge


class CustomBayesianRidge(BayesianRidge):
    def __init__(self, n_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6, compute_score=False,
                 fit_intercept=True, copy_X=True, verbose=False):
        super().__init__(n_iter=n_iter, tol=tol, alpha_1=alpha_1, alpha_2=alpha_2,
                         lambda_1=lambda_1, lambda_2=lambda_2, compute_score=compute_score,
                         fit_intercept=fit_intercept, copy_X=copy_X, verbose=verbose)


if __name__ == '__main__':
    train_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_train.csv'
    test_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_test.csv'
    df_train = load_data(train_file_path)
    df_test = load_data(test_file_path)
    df_train_x, df_train_y = split_features_labels(df_train)
    df_test_x, df_test_y = split_features_labels(df_test)

    # Bayesian Regression Hyperparameter Search Space (Only Two Parameters Explored)
    param_grid = {
        'alpha_1': np.logspace(-6, -2, 10).tolist(),
        'lambda_1': np.logspace(-6, -2, 10).tolist()
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
    best_params, best_metrics = search_hyperparameters(CustomBayesianRidge, param_grid, df_train_x, df_train_y,
                                                       feature_engineering_params=feature_engineering_params3)

    # Evaluate the best model
    metrics = model_evaluation(CustomBayesianRidge(**best_params), df_train_x, df_train_y, df_test_x, df_test_y,
                               feature_engineering_params=feature_engineering_params3)