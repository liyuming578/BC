from src.data_preprocess import load_data, split_features_labels
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

if __name__ == '__main__':
    # File Path
    train_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_train.csv'
    test_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_test.csv'
    save_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\results\\evaluation_results.csv'

    # load data
    df_train = load_data(train_file_path)
    df_test = load_data(test_file_path)

    # split feature and label
    df_train_x, df_train_y = split_features_labels(df_train)
    df_test_x, df_test_y = split_features_labels(df_test)

    # Calculate the mean of the training set labels
    mean_value = df_train_y.mean()

    # Use the mean of the training set as the predicted value
    y_train_pred = np.full(df_train_y.shape, mean_value)
    y_test_pred = np.full(df_test_y.shape, mean_value)  # 生成与df_test_y相同长度的数组

    # Calculate the error on the training set
    train_mae = mean_absolute_error(df_train_y, y_train_pred)
    train_mse = mean_squared_error(df_train_y, y_train_pred)
    train_rmse = np.sqrt(train_mse)

    # Calculate the error on the test set
    test_mae = mean_absolute_error(df_test_y, y_test_pred)
    test_mse = mean_squared_error(df_test_y, y_test_pred)
    test_rmse = np.sqrt(test_mse)

    # summarize the results
    metrics = {
        'train_mae': train_mae,
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'test_mae': test_mae,
        'test_mse': test_mse,
        'test_rmse': test_rmse
    }

    # Save the results as a CSV file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(save_path, index=False)

    # Output
    print(metrics)