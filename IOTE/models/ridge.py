import numpy as np
from src.data_preprocess import load_data, split_features_labels
from src.model_evaluation import search_hyperparameters, model_evaluation
from sklearn.linear_model import Ridge


class CustomRidge(Ridge):
    def __init__(self, alpha=1.0, fit_intercept=True, copy_X=True, max_iter=None,
                 tol=0.001, solver='auto', random_state=None):
        # Call the constructor of the parent class Ridge through super()
        super().__init__(alpha=alpha, fit_intercept=fit_intercept, copy_X=copy_X,
                         max_iter=max_iter, tol=tol, solver=solver, random_state=random_state)


if __name__ == '__main__':
    train_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_train.csv'
    test_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_test.csv'
    df_train = load_data(train_file_path)
    df_test = load_data(test_file_path)
    df_train_x, df_train_y = split_features_labels(df_train)
    df_test_x, df_test_y = split_features_labels(df_test)

    param_grid = {
        'alpha': np.logspace(-4, 2, 20).tolist()  # Generate 20 values from 10^-4 to 10^2
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

    # Finding the best hyperparameters
    best_params, best_metrics = search_hyperparameters(CustomRidge, param_grid, df_train_x, df_train_y,
                                                       feature_engineering_params=feature_engineering_params3)

    # Evaluating the Model
    metrics = model_evaluation(CustomRidge(**best_params), df_train_x, df_train_y, df_test_x, df_test_y,
                               feature_engineering_params=feature_engineering_params3)
