import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import euclidean_distances
from src.data_preprocess import load_data, split_features_labels
from src.model_evaluation import search_hyperparameters, model_evaluation


class RBFNetwork(BaseEstimator, RegressorMixin):
    def __init__(self, gamma_prime=1.0, k=10, alpha=1.0, n_init='auto'):
        self.gamma_prime = gamma_prime  # Hyperparameter gamma'
        self.k = k  # Number of RBF centers (number of clusters in K-means)
        self.alpha = alpha  # Regularization parameter for Ridge regression
        self.n_init = n_init  # n_init parameter for KMeans
        self.kmeans = None
        self.gamma_d = None
        self.gamma = None
        self.centers = None
        self.ridge = Ridge(alpha=self.alpha)

    def _calculate_gamma_d(self, X):
        # Calculate the width of each dimension of the feature space
        delta_x = np.ptp(X, axis=0)  # ptp is max - min, returns the range of each feature
        # Calculate the average spacing alpha
        alpha = np.mean(delta_x) / (self.k ** (1 / X.shape[1]))
        # Calculate gamma_d
        gamma_d = 1 / (8 * alpha ** 2)
        return gamma_d

    def _rbf_transform(self, X):
        # Convert input data into RBF kernel output
        distances = euclidean_distances(X, self.centers)
        return np.exp(-self.gamma * (distances ** 2))

    def fit(self, X, y):
        # Make sure X is a numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Use K-means to select RBF centers and set n_init and init parameters
        self.kmeans = KMeans(n_clusters=self.k, n_init=self.n_init, init='random', random_state=42)
        self.kmeans.fit(X)
        self.centers = self.kmeans.cluster_centers_  # Get cluster centers

        # Calculate gamma_d and set gamma
        self.gamma_d = self._calculate_gamma_d(X)
        self.gamma = self.gamma_prime * self.gamma_d

        # RBF Kernel Transformation
        X_rbf = self._rbf_transform(X)

        # Training the model using Ridge regression
        self.ridge.fit(X_rbf, y)
        return self

    def predict(self, X):
        # 确保X是numpy数组
        if isinstance(X, pd.DataFrame):
            X = X.values

        # RBF Kernel Transformation
        X_rbf = self._rbf_transform(X)

        # predict
        return self.ridge.predict(X_rbf)

    def get_params(self, deep=True):
        return {"gamma_prime": self.gamma_prime, "k": self.k, "alpha": self.alpha, "n_init": self.n_init}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "3"
    # load data
    train_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_train.csv'
    test_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_test.csv'
    df_train = load_data(train_file_path)
    df_test = load_data(test_file_path)
    df_train_x, df_train_y = split_features_labels(df_train)
    df_test_x, df_test_y = split_features_labels(df_test)

    # RBFNetwork Key parameter search space
    param_grid = {
        'gamma_prime': [0.01, 0.1, 1, 10, 100],  # Choose the value of γ' on the log scale
        'k': list(np.arange(10, 100, 10))  # k is between 10 and 100 in steps of 10
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
    best_params, best_metrics = search_hyperparameters(RBFNetwork, param_grid, df_train_x, df_train_y,
                                                       feature_engineering_params=feature_engineering_params3)

    # Make sure k is an integer
    best_params = {key: int(value) if key == 'k' else value for key, value in best_params.items()}

    # Evaluate the best model
    metrics = model_evaluation(RBFNetwork(**best_params), df_train_x, df_train_y, df_test_x, df_test_y,
                               feature_engineering_params=feature_engineering_params3)
