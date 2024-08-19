import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_data(file_path):
    df = pd.read_csv(file_path, header=2)
    df.dropna(axis=1, how='all', inplace=True)
    if 'SubjectID' in df.columns:
        df.drop('SubjectID', axis=1, inplace=True)
    else:
        print('SubjectID does not exist')

    # Keep only columns whose names contain "_1", "_2", "_3" or "_4"
    features = list(
        set([col.rsplit('_', 1)[0] for col in df.columns if any(suffix in col for suffix in ['_1', '_2', '_3', '_4'])]))

    # Traverse each row of data
    for idx in range(len(df)):

        # Iterate over each feature
        for feature in features:
            rounds = [col for col in df.columns if col.startswith(feature)]

            # Extract all rounds of data for this feature
            round_data = df.loc[idx, rounds]

            # Check whether there are null values in each round of data for this feature
            if round_data.isnull().any():
                # Calculate the average of other rounds
                fill_value = round_data.mean()

                for col in rounds:
                    if pd.isnull(df.loc[idx, col]):
                        df.loc[idx, col] = fill_value

    # Create a new DataFrame to store the merged data
    new_df = pd.DataFrame()

    # Iterate over each feature and merge them
    for feature in features:
        rounds = [col for col in df.columns if col.startswith(feature)]
        # Calculate the average of these four columns and create a new column
        new_df[feature] = df[rounds].mean(axis=1)

    # Add additional columns not in features to new_df
    other_columns = [col for col in df.columns if not any(suffix in col for suffix in ['_1', '_2', '_3', '_4'])]
    new_df[other_columns] = df[other_columns]

    # First get the numerical features
    numeric_features = new_df.select_dtypes(include=[np.number]).columns
    categorical_features = new_df.select_dtypes(include=['object', 'category']).columns

    categorical_columns = ['Gender', 'Age', 'Ethnicity']
    expected_columns = [
        'Gender_Female', 'Gender_Male',
        'Age_18-20', 'Age_21-25', 'Age_21-30', 'Age_26-30',
        'Age_31-40', 'Age_41-50', 'Age_51-60', 'Age_>60',
        'Ethnicity_American Indian or Alaskan Native',
        'Ethnicity_Asian', 'Ethnicity_Black or African-American',
        'Ethnicity_Hispanic/Latino', 'Ethnicity_Multiracial', 'Ethnicity_White'
    ]
    # One-hot encoding of categorical features
    ohe_df = pd.get_dummies(new_df[categorical_features], columns=categorical_columns)  # one hot coding dataframe
    # Find missing columns
    missing_columns = set(expected_columns) - set(ohe_df.columns)
    # Complete the missing columns and fill them with False
    for col in missing_columns:
        ohe_df[col] = False

    # Make sure the order of the columns matches the expected column order
    ohe_df = ohe_df.reindex(columns=expected_columns)

    # Finally, merge the datasets, keeping the order of numerical feature columns and Boolean feature columns unchanged.
    final_df = pd.concat([new_df[numeric_features], ohe_df], axis=1)

    return final_df


def split_features_labels(df, label_columns='aveOralM'):
    """
    Separate features and labels from a dataset.

    Parameters:
    df (pd.DataFrame): Dataset containing features and labels.
    label_columns (list): List containing label column names.

    Returns:
    tuple: (features_df, labels_df)
    features_df (pd.DataFrame): DataFrame containing only features.
    labels_df (pd.DataFrame): DataFrame containing only labels.
    """
    # Extract label column
    labels_df = df[label_columns]

    # The remaining columns are feature columns
    features_df = df.drop(columns=label_columns)

    return features_df, labels_df


def fill_missing_values(X_train_df, X_test_df, strategy='mean'):
    """
    Fill the empty values of the numerical features of the training set and the test set according to the statistical information of the training set, and merge them back into the original dataset.

    parameters:
    - X_train_df: feature of train set (pandas DataFrame)
    - X_test_df: feature of test set (pandas DataFrame)
    - strategy: imputation strategy ('mean', 'median', 'most_frequent')
        - 'mean': Filling the missing values with the mean
        - 'median': Filling empty values using median
        - 'most_frequent': Filling empty values with majority

    return:
    - X_train_filled_df: Training set after filling in the empty values (pandas DataFrame)
    - X_test_filled_df: Test set after filling in the empty values (pandas DataFrame)
    """
    # Selecting Numeric and Boolean Features
    numeric_features = X_train_df.select_dtypes(include=[np.number]).columns
    bool_features = X_train_df.select_dtypes(include=[bool]).columns

    # Fill only numerical features
    imputer = SimpleImputer(strategy=strategy)

    # Fill missing values of numerical features in the training set
    X_train_filled_numeric = imputer.fit_transform(X_train_df[numeric_features])

    # Fill in the numerical features of the test set with the fill-in values of the training set
    X_test_filled_numeric = imputer.transform(X_test_df[numeric_features])

    # Convert the padded numerical features to a DataFrame, keeping the original column names and row indices
    X_train_filled_numeric_df = pd.DataFrame(X_train_filled_numeric, columns=numeric_features, index=X_train_df.index)
    X_test_filled_numeric_df = pd.DataFrame(X_test_filled_numeric, columns=numeric_features, index=X_test_df.index)

    # Merge the padded numerical features and the original Boolean features back into the dataset
    X_train_filled_df = pd.concat([X_train_filled_numeric_df, X_train_df[bool_features]], axis=1)
    X_test_filled_df = pd.concat([X_test_filled_numeric_df, X_test_df[bool_features]], axis=1)

    return X_train_filled_df, X_test_filled_df


def standardize_datasets(X_train_df, X_test_df):
    """
    Standardize the numerical features of the training set and test set according to the statistical information of the training set, and merge them back into the original dataset.

    Parameters:
    - X_train_df: training set features (pandas DataFrame)
    - X_test_df: test set features (pandas DataFrame)

    Returns:
    - X_train_standardized_df: standardized training set (pandas DataFrame)
    - X_test_standardized_df: standardized test set (pandas DataFrame)
    """
    # Selecting Numeric and Boolean Features
    numeric_features = X_train_df.select_dtypes(include=[np.number]).columns
    bool_features = X_train_df.select_dtypes(include=[bool]).columns

    # Normalize only numerical features
    scaler = StandardScaler()

    # Standardize the numerical features of the training set
    X_train_standardized_numeric = scaler.fit_transform(X_train_df[numeric_features])

    # Use the normalization parameters of the training set to normalize the numerical features of the test set
    X_test_standardized_numeric = scaler.transform(X_test_df[numeric_features])

    # Convert the standardized numerical features to DataFrame, keeping the original column names and row indices
    X_train_standardized_numeric_df = pd.DataFrame(X_train_standardized_numeric, columns=numeric_features, index=X_train_df.index)
    X_test_standardized_numeric_df = pd.DataFrame(X_test_standardized_numeric, columns=numeric_features, index=X_test_df.index)

    # Merge the normalized numerical features and the original Boolean features back into the dataset
    X_train_standardized_df = pd.concat([X_train_standardized_numeric_df, X_train_df[bool_features]], axis=1)
    X_test_standardized_df = pd.concat([X_test_standardized_numeric_df, X_test_df[bool_features]], axis=1)

    return X_train_standardized_df, X_test_standardized_df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_train.csv'
    test_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_test.csv'
    df_train = load_data(train_file_path)
    df_test = load_data(test_file_path)
    df_train_x, df_train_y = split_features_labels(df_train)
    df_test_x, df_test_y = split_features_labels(df_test)

    df_train_x, df_test_x = fill_missing_values(df_train_x, df_test_x)
    df_train_x, df_test_x = standardize_datasets(df_train_x, df_test_x)

    print(len(df_train_x), len(df_train_y))

    # df_train_x.to_csv('train_x.csv', index=False)
    # df_test_x.to_csv('test_x.csv', index=False)
    # df_train.to_csv('train.csv', index=False)
    # df_test.to_csv('test.csv', index=False)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
