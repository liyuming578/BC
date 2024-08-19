from src.data_preprocess import load_data, split_features_labels
from src.model_evaluation import model_evaluation
from sklearn.neighbors import KNeighborsRegressor

if __name__ == '__main__':
    # load data
    train_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_train.csv'
    test_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_test.csv'

    df_train = load_data(train_file_path)
    df_test = load_data(test_file_path)

    # split feature and label
    df_train_x, df_train_y = split_features_labels(df_train)
    df_test_x, df_test_y = split_features_labels(df_test)

    # Initializing the 1NN model
    model = KNeighborsRegressor(n_neighbors=1)

    # Evaluating the Model
    metrics = model_evaluation(model, df_train_x, df_train_y, df_test_x, df_test_y)

    # output results
    print(metrics)