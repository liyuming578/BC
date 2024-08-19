import numpy as np
from src.data_preprocess import load_data, split_features_labels
from src.model_evaluation import search_hyperparameters, model_evaluation
from sklearn.svm import SVR


class CustomSVR(SVR):
    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', degree=3, gamma='scale', coef0=0.0,
                 tol=0.001, shrinking=True, cache_size=200, verbose=False, max_iter=-1):
        # Call the constructor of the parent class SVR through super()
        super().__init__(C=C, epsilon=epsilon, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
                         tol=tol, shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=max_iter)


if __name__ == '__main__':
    train_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_train.csv'
    test_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_test.csv'
    df_train = load_data(train_file_path)
    df_test = load_data(test_file_path)
    df_train_x, df_train_y = split_features_labels(df_train)
    df_test_x, df_test_y = split_features_labels(df_test)

    # Hyperparameter Ranges
    param_grid = {
        'C': np.logspace(-3, 3, 10).tolist(),
        'epsilon': np.logspace(-4, 1, 10).tolist()
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
    best_params, best_metrics = search_hyperparameters(CustomSVR, param_grid, df_train_x, df_train_y,
                                                       feature_engineering_params=feature_engineering_params3)

    # Evaluating the Model
    metrics = model_evaluation(CustomSVR(**best_params), df_train_x, df_train_y, df_test_x, df_test_y,
                               feature_engineering_params=feature_engineering_params3)
