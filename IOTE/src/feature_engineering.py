import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA


def select_features_pearson(X_train_df, X_test_df, y_train, threshold=0.7):
    """
    Use the Pearson correlation coefficient to select numerical features that are correlated with the target variable above a specified threshold, and merge the boolean features back into the training and test sets.

    Parameters:
    - X_train_df: training set features (pandas DataFrame)
    - X_test_df: test set features (pandas DataFrame)
    - y_train: training set target variable (pandas Series)
    - threshold: selected correlation coefficient threshold (float)

    Returns:
    - X_train_selected_df: selected training set (pandas DataFrame)
    - X_test_selected_df: selected test set (pandas DataFrame)
    """
    # Select numerical features and Boolean features
    numeric_features = X_train_df.select_dtypes(include=[np.number]).columns
    bool_features = X_train_df.select_dtypes(include=[bool]).columns

    # Calculate the Pearson correlation coefficient between each numerical feature and the target variable
    selected_numeric_features = []
    for feature in numeric_features:
        corr = np.corrcoef(X_train_df[feature], y_train)[0, 1]
        if abs(corr) >= threshold:  # Select features with correlation with target variable exceeding threshold
            selected_numeric_features.append(feature)

    # Select these features from the training and test sets and merge the Boolean features back
    X_train_selected_df = pd.concat([X_train_df[selected_numeric_features], X_train_df[bool_features]], axis=1)
    X_test_selected_df = pd.concat([X_test_df[selected_numeric_features], X_test_df[bool_features]], axis=1)

    return X_train_selected_df, X_test_selected_df


def apply_pca(X_train_df, X_test_df, n_components=0.95):
    """
    Use PCA to reduce the dimension of numerical features and apply it to the training set and test set.

    Parameters:
    - X_train_df: training set features (pandas DataFrame)
    - X_test_df: test set features (pandas DataFrame)
    - n_components: number of target components or variance explained by PCA (float or int)

    Returns:
    - X_train_pca_df: training set after PCA dimension reduction (pandas DataFrame)
    - X_test_pca_df: test set after PCA dimension reduction (pandas DataFrame)
    """
    # Selecting Numeric and Boolean Features
    numeric_features = X_train_df.select_dtypes(include=[np.number]).columns
    bool_features = X_train_df.select_dtypes(include=[bool]).columns

    # Dimensionality reduction using PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_df[numeric_features])
    X_test_pca = pca.transform(X_test_df[numeric_features])

    # Convert the reduced features into a DataFrame and keep the original row index
    pca_feature_names = [f'PCA_{i + 1}' for i in range(X_train_pca.shape[1])]
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_feature_names, index=X_train_df.index)
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_feature_names, index=X_test_df.index)

    # Merge Boolean features back into the dataset
    X_train_pca_df = pd.concat([X_train_pca_df, X_train_df[bool_features]], axis=1)
    X_test_pca_df = pd.concat([X_test_pca_df, X_test_df[bool_features]], axis=1)

    return X_train_pca_df, X_test_pca_df


def apply_kpca(X_train_df, X_test_df, n_components=20, gamma=0.1):
    """
    Apply Kernel PCA to the input features for dimensionality reduction.

    Parameters:
    - X_train_df: training set features (pandas DataFrame)
    - X_test_df: test set features (pandas DataFrame)
    - n_components: number of components for Kernel PCA (int)
    - gamma: parameter of RBF kernel (float)

    Returns:
    - X_train_kpca_df: training set after dimensionality reduction (pandas DataFrame)
    - X_test_kpca_df: test set after dimensionality reduction (pandas DataFrame)
    """
    # Selecting Numeric and Boolean Features
    numeric_features = X_train_df.select_dtypes(include=[np.number]).columns
    bool_features = X_train_df.select_dtypes(include=[bool]).columns

    # Dimensionality reduction using Kernel PCA
    kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma)
    X_train_kpca = kpca.fit_transform(X_train_df[numeric_features])
    X_test_kpca = kpca.transform(X_test_df[numeric_features])

    # Convert the reduced features into a DataFrame and keep the original row index
    kpca_feature_names = [f'KPCA_{i + 1}' for i in range(X_train_kpca.shape[1])]
    X_train_kpca_df = pd.DataFrame(X_train_kpca, columns=kpca_feature_names, index=X_train_df.index)
    X_test_kpca_df = pd.DataFrame(X_test_kpca, columns=kpca_feature_names, index=X_test_df.index)

    # Merge Boolean features back into the dataset
    X_train_kpca_df = pd.concat([X_train_kpca_df, X_train_df[bool_features]], axis=1)
    X_test_kpca_df = pd.concat([X_test_kpca_df, X_test_df[bool_features]], axis=1)

    return X_train_kpca_df, X_test_kpca_df
