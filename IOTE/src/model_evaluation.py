import numpy as np
import itertools
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data_preprocess import fill_missing_values, standardize_datasets
from src.visulization import show
from src.feature_engineering import select_features_pearson, apply_pca, apply_kpca

"""
# Using PCA as a feature engineering strategy
feature_engineering_params = {
    'strategy': 'pca',
    'n_components': 0.95
}
metrics = cross_validate(model, X_train_df, y_train_df, n_splits=5, feature_engineering_params=feature_engineering_params)

# Using Pearson correlation coefficient to select features
feature_engineering_params = {
    'strategy': 'pearson',
    'threshold': 0.7
}
metrics = cross_validate(model, X_train_df, y_train_df, n_splits=5, feature_engineering_params=feature_engineering_params)

# Using polynomial feature generation
feature_engineering_params = {
    'strategy': 'poly',
    'degree': 2,
    'n_components': 46,
    'gamma': 0.1
}
metrics = cross_validate(model, X_train_df, y_train_df, n_splits=5, feature_engineering_params=feature_engineering_params)
"""


def cross_validate(model, X_train_df, y_train_df, n_splits=5, impute_strategy='mean', feature_engineering_params=None):
    """
    Evaluate model performance using cross validation and output MAE, MSE and RMSE for each fold and their average.

    Parameters:
    - model: Model to be evaluated (models that conform to the scikit-learn API)
    - X_train_df: Training set features (pandas DataFrame)
    - y_train_df: Training set target variable (pandas DataFrame or Series)
    - n_splits: Number of cross validation folds (default is 5)
    - impute_strategy: Strategy for imputing empty values ('mean', 'median', 'most_frequent'), default is 'mean'
    - feature_engineering_params: Dictionary containing feature engineering strategies and parameters, default is None

    Returns:
    - mae_list: List of MAE for each fold
    - mae_avg: Average of MAE
    - mse_list: List of MSE for each fold
    - mse_avg: MSE The average value of
    - rmse_list: RMSE list of each fold
    - rmse_avg: The average value of RMSE
    """
    # Initializing KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initializes a list to store errors
    mae_list = []
    mse_list = []
    rmse_list = []

    # Cross Validation
    for train_index, val_index in kf.split(X_train_df):
        # Split into training set and validation set
        X_train_df_fold, X_val_df_fold = X_train_df.iloc[train_index], X_train_df.iloc[val_index]
        y_train_df_fold, y_val_df_fold = y_train_df.iloc[train_index], y_train_df.iloc[val_index]

        # Handling missing values
        X_train_df_fold, X_val_df_fold = fill_missing_values(X_train_df_fold, X_val_df_fold, strategy=impute_strategy)

        # Standardizing Data
        X_train_df_fold, X_val_df_fold = standardize_datasets(X_train_df_fold, X_val_df_fold)

        # Handling feature engineering parameter dictionaries
        if feature_engineering_params:
            strategy = feature_engineering_params.get('strategy')

            if strategy == 'pearson':
                threshold = feature_engineering_params.get('threshold', 0.7)
                X_train_df_fold, X_val_df_fold = select_features_pearson(X_train_df_fold, X_val_df_fold,
                                                                         y_train_df_fold, threshold=threshold)

            elif strategy == 'pca':
                n_components = feature_engineering_params.get('n_components', 0.95)
                X_train_df_fold, X_val_df_fold = apply_pca(X_train_df_fold, X_val_df_fold, n_components=n_components)


            elif strategy == 'kpca':

                n_components = feature_engineering_params.get('n_components', 20)

                gamma = feature_engineering_params.get('gamma', 0.1)


                X_train_df_fold, X_val_df_fold = apply_kpca(

                    X_train_df_fold, X_val_df_fold, n_components=n_components, gamma=gamma

                )

        # train the model
        model.fit(X_train_df_fold, y_train_df_fold)

        # predict
        y_pred = model.predict(X_val_df_fold)

        # Convert y_val_df to a NumPy array and convert it to a one-dimensional array
        y_val = y_val_df_fold.to_numpy().ravel()

        # calculate error
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)

        # add the error in the list
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)

    # calculate mean error
    mae_avg = np.mean(mae_list)
    mse_avg = np.mean(mse_list)
    rmse_avg = np.mean(rmse_list)

    # summarize results
    metrics = {
        'mae_avg': mae_avg,
        'mse_avg': mse_avg,
        'rmse_avg': rmse_avg
    }

    return metrics


def search_hyperparameters(model_class, param_grid, X_train_df, y_train_df, n_splits=5, impute_strategy='mean',
                           feature_engineering_params=None, plot_results=True, save_path_results=True,
                           save_path_figure=True):
    """
    Search for the best hyperparameter combination and evaluate its performance using cross validation.

    Parameters:
    - model_class: the class of the machine learning model
    - param_grid: hyperparameter search space (in dictionary form)
    - X_train_df: training set features (pandas DataFrame)
    - y_train_df: training set target variable (pandas DataFrame or Series)
    - n_splits: the number of cross validation folds (default is 5)
    - impute_strategy: strategy for filling empty values ​​('mean', 'median', 'most_frequent'), default is 'mean'
    - plot_results: whether to draw a graph of the relationship between hyperparameters and loss (default is False)

    Returns:
    - best_params: the best hyperparameter combination
    - best_metrics: the cross validation performance metric corresponding to the best hyperparameter combination
    """
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())

    results = []

    for param_values in param_combinations:
        params = dict(zip(param_names, param_values))
        model = model_class(**params)
        metrics = cross_validate(model, X_train_df, y_train_df, n_splits=n_splits, impute_strategy=impute_strategy,
                                 feature_engineering_params=feature_engineering_params)
        result = {**params, **metrics}
        results.append(result)
        print(
            f"Tested {params}: MAE={metrics['mae_avg']:.4f}, MSE={metrics['mse_avg']:.4f}, RMSE={metrics['rmse_avg']:.4f}")

    # convert results to DataFrame
    results_df = pd.DataFrame(results)
    if save_path_results:
        results_df.to_csv('D:\\Users\\allmi\\PycharmProjects\\IOTE\\results\\hyperparameters_results.csv', index=False)
    # choose the min MAE combination
    best_index = results_df['mae_avg'].idxmin()
    best_params = results_df.loc[best_index, param_names].to_dict()
    best_metrics = results_df.loc[best_index, ['mae_avg', 'mse_avg', 'rmse_avg']].to_dict()

    if plot_results:
        show(param_names, results_df, save_path_figure)

    print(f"\nBest parameters: {best_params}")
    print(f"Best cross-validation MAE: {best_metrics['mae_avg']:.4f}")

    return best_params, best_metrics


def model_evaluation(model, X_train_df, y_train_df, X_test_df, y_test_df, impute_strategy='mean',
                     feature_engineering_params=None, save_path=True):
    """
    Train the given model and evaluate its performance on the training and test sets.

    Parameters:
    - model: the model to be evaluated (a model that conforms to the scikit-learn API)
    - X_train_df: training set features (pandas DataFrame)
    - y_train_df: training set target variable (pandas DataFrame or Series)
    - X_test_df: test set features (pandas DataFrame)
    - y_test_df: test set target variable (pandas DataFrame or Series)
    - impute_strategy: strategy for imputing empty values ('mean', 'median', 'most_frequent'), default is 'mean'
    - feature_engineering_params: dictionary containing feature engineering strategies and parameters, default is None

    Returns:
    - metrics: dictionary containing MAE, MSE, RMSE of the training and test sets
    """

    X_train_df, X_test_df = fill_missing_values(X_train_df, X_test_df, strategy=impute_strategy)

    X_train_df, X_test_df = standardize_datasets(X_train_df, X_test_df)

    if feature_engineering_params:
        strategy = feature_engineering_params.get('strategy')

        if strategy == 'pearson':
            threshold = feature_engineering_params.get('threshold', 0.7)
            X_train_df, X_test_df = select_features_pearson(X_train_df, X_test_df, y_train_df, threshold=threshold)

        elif strategy == 'pca':
            n_components = feature_engineering_params.get('n_components', 0.95)
            X_train_df, X_test_df = apply_pca(X_train_df, X_test_df, n_components=n_components)


        elif strategy == 'kpca':

            n_components = feature_engineering_params.get('n_components', 20)

            gamma = feature_engineering_params.get('gamma', 0.1)


            X_train_df, X_test_df = apply_kpca(

                X_train_df, X_test_df, n_components=n_components, gamma=gamma

            )

    # train the model
    model.fit(X_train_df, y_train_df)

    # predict the train set
    y_train_pred = model.predict(X_train_df)

    # predict the test set
    y_test_pred = model.predict(X_test_df)

    # Convert y_train_df and y_test_df to NumPy arrays and ensure the dimensions are consistent
    y_train = y_train_df.to_numpy().ravel()
    y_test = y_test_df.to_numpy().ravel()

    # Calculate the error on the training set
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)

    # Calculate the error on the test set
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)

    # Summarize the results
    metrics = {
        'train_mae': train_mae,
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'test_mae': test_mae,
        'test_mse': test_mse,
        'test_rmse': test_rmse
    }

    metrics_df = pd.DataFrame([metrics])
    if save_path:
        metrics_df.to_csv('D:\\Users\\allmi\\PycharmProjects\\IOTE\\results\\evaluation_results.csv', index=False)

    # 输出结果
    print(f"Training set - MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}")
    print(f"Test set - MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}")

    return metrics
