import numpy as np
from src.data_preprocess import load_data, split_features_labels
from src.model_evaluation import search_hyperparameters, model_evaluation
from sklearn.neighbors import KNeighborsRegressor


class CustomKNN(KNeighborsRegressor):
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                 metric_params=None, n_jobs=None):
        # Call the constructor of the parent class KNeighborsRegressor through super()
        super().__init__(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size,
                         p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs)


if __name__ == '__main__':
    train_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_train.csv'
    test_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_test.csv'
    df_train = load_data(train_file_path)
    df_test = load_data(test_file_path)
    df_train_x, df_train_y = split_features_labels(df_train)
    df_test_x, df_test_y = split_features_labels(df_test)

    param_grid = {
        'n_neighbors': np.arange(1, 31).tolist(),  # Test k values from 1 to 30
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

    best_params, best_metrics = search_hyperparameters(CustomKNN, param_grid, df_train_x, df_train_y,
                                                       feature_engineering_params=feature_engineering_params3)

    metrics = model_evaluation(CustomKNN(**{key: int(value) if key == 'n_neighbors' else value
                                            for key, value in best_params.items()}), df_train_x, df_train_y, df_test_x,
                               df_test_y, feature_engineering_params=feature_engineering_params3)
