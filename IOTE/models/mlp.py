import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.data_preprocess import load_data, split_features_labels, fill_missing_values
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


class MLP(nn.Module):
    def __init__(self, input_size=10, hidden_size1=128, hidden_size2=64, hidden_size3=32, output_size=1, dropout_rate=0.1):
        super(MLP, self).__init__()
        # First hidden layer with BatchNorm and Dropout
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second hidden layer with BatchNorm and Dropout
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Third hidden layer with BatchNorm and Dropout
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Output layer
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        # Input -> First hidden layer with ReLU, BatchNorm, and Dropout
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        # First hidden layer -> Second hidden layer with ReLU, BatchNorm, and Dropout
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        # Second hidden layer -> Third hidden layer with ReLU, BatchNorm, and Dropout
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        # Third hidden layer -> Output layer
        x = self.fc4(x)
        return x

def df_to_tensor(df):
    # Use to_numpy() to convert the DataFrame to a numpy floating point array and then to a Tensor
    return torch.tensor(df.to_numpy(dtype=float), dtype=torch.float32)


def calculate_metrics(outputs, targets):
    # Calculate MAE, MSE, and RMSE
    mae = torch.mean(torch.abs(outputs - targets)).item()
    mse = torch.mean((outputs - targets) ** 2).item()
    rmse = torch.sqrt(torch.mean((outputs - targets) ** 2)).item()
    return mae, mse, rmse


def validation(model, optimizer, criterion, scheduler, df_train_x, df_train_y, num_epochs=50, val_split=0.2, batch_size=10):
    # Split the dataset into training and validation sets
    train_indices, val_indices = train_test_split(range(len(df_train_x)), test_size=val_split, random_state=42)

    # Create training and validation DataFrames using the indices
    df_train_x_split = df_train_x.iloc[train_indices]
    df_val_x_split = df_train_x.iloc[val_indices]
    df_train_y_split = df_train_y.iloc[train_indices]
    df_val_y_split = df_train_y.iloc[val_indices]

    # Fill missing values in the training and validation sets
    df_train_x_filled, df_val_x_filled = fill_missing_values(df_train_x_split, df_val_x_split)

    # Convert the filled DataFrames to Tensors
    train_data = df_to_tensor(df_train_x_filled)
    train_targets = df_to_tensor(df_train_y_split).unsqueeze(1)  # Add an extra dimension to match model output
    val_data = df_to_tensor(df_val_x_filled)
    val_targets = df_to_tensor(df_val_y_split).unsqueeze(1)

    # Create the complete datasets
    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)

    # Create DataLoader for both training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_mae_losses = []  # To store MAE losses for training set
    val_mae_losses = []  # To store MAE losses for validation set

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_mae = 0.0
        running_mse = 0.0
        running_rmse = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            # Compute training metrics
            mae, mse, rmse = calculate_metrics(outputs, targets)
            running_mae += mae
            running_mse += mse
            running_rmse += rmse

        # Step the learning rate scheduler
        scheduler.step()

        # Calculate average training loss
        train_mae_avg = running_mae / len(train_loader)
        train_mae_losses.append(train_mae_avg)

        # Evaluate model on validation set
        model.eval()  # Set model to evaluation mode
        val_mae = 0.0
        val_mse = 0.0
        val_rmse = 0.0

        with torch.no_grad():  # Disable gradient computation
            for inputs, targets in val_loader:
                outputs = model(inputs)
                mae, mse, rmse = calculate_metrics(outputs, targets)
                val_mae += mae
                val_mse += mse
                val_rmse += rmse

        # Calculate average validation loss
        val_mae_avg = val_mae / len(val_loader)
        val_mse_avg = val_mse / len(val_loader)
        val_rmse_avg = val_rmse / len(val_loader)
        val_mae_losses.append(val_mae_avg)

        # Print training and validation loss for the current epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training MAE: {train_mae_avg:.4f}, Validation MAE: {val_mae_avg:.4f}')

    # Summarize final validation metrics
    metrics = {
        'mae_avg': val_mae_avg,
        'mse_avg': val_mse_avg,
        'rmse_avg': val_rmse_avg
    }

    print("Final validation metrics:", metrics)
    return metrics


def hyperparameter_tuning(df_train_x, df_train_y, criterion, hyperparams, num_epochs=50, val_split=0.2, batch_size=10):
    best_metrics = None
    best_params = None
    all_results = []  # List to store the results of all hyperparameter combinations

    for opt_name, params in hyperparams.items():
        print(f"Training with {opt_name} optimizer and learning rate {params['lr']}")

        # Initialize the model
        model = MLP(input_size=df_train_x.shape[1])

        # Select the appropriate optimizer based on the optimizer name
        if opt_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=params['lr'])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)  # 学习率每10个epoch降低为原来的0.1倍
            momentum = None  # Adam does not use momentum in the same way as SGD
        elif opt_name == 'SGD':
            momentum = params.get('momentum', None)
            optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=momentum)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)  # 学习率每10个epoch降低为原来的0.1倍

        # Perform validation and obtain results
        metrics = validation(model, optimizer, criterion, scheduler, df_train_x, df_train_y,
                             num_epochs=num_epochs, val_split=val_split, batch_size=batch_size)

        # Update the best hyperparameters and corresponding results
        if best_metrics is None or metrics['mae_avg'] < best_metrics['mae_avg']:
            best_metrics = metrics
            best_params = {'optimizer': opt_name, 'learning_rate': params['lr'], 'momentum': momentum}

        # Store the current result in the all_results list
        result = {
            'optimizer': opt_name,
            'learning_rate': params['lr'],
            'momentum': momentum,
            'mae_avg': metrics['mae_avg'],
            'mse_avg': metrics['mse_avg'],
            'rmse_avg': metrics['rmse_avg']
        }
        all_results.append(result)

    # Save all results to a CSV file
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('hyperparameter_tuning_results.csv', index=False)

    print("Best hyperparameters found:")
    print(f"Optimizer: {best_params['optimizer']}, Learning Rate: {best_params['learning_rate']}, Momentum: {best_params['momentum']}")
    print("Validation metrics with best hyperparameters:", best_metrics)

    return best_params, best_metrics


def evaluation(model, best_params, criterion, df_train_x, df_train_y, df_test_x, df_test_y, num_epochs=50,
               batch_size=10):
    # Fill missing values in the training and testing sets
    df_train_x_filled, df_test_x_filled = fill_missing_values(df_train_x, df_test_x)

    # Convert the filled DataFrames to Tensors
    train_data = df_to_tensor(df_train_x_filled)
    train_targets = df_to_tensor(df_train_y).unsqueeze(1)  # Add an extra dimension to match model output
    test_data = df_to_tensor(df_test_x_filled)
    test_targets = df_to_tensor(df_test_y).unsqueeze(1)

    # Create the complete datasets
    train_dataset = TensorDataset(train_data, train_targets)
    test_dataset = TensorDataset(test_data, test_targets)

    # Create DataLoader for both training and testing sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize optimizer with the best hyperparameters
    optimizer = None
    scheduler = None
    if best_params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    elif best_params['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=best_params['learning_rate'], momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # Lists to store training and testing losses
    train_losses = []
    test_losses = []

    # Lists to store training and testing metrics
    train_mae_list = []
    train_mse_list = []
    train_rmse_list = []

    test_mae_list = []
    test_mse_list = []
    test_rmse_list = []

    # Re-train the model on the entire training dataset
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        train_mae = 0.0
        train_mse = 0.0
        train_rmse = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss

            if torch.isnan(loss).any():
                raise ValueError(f"NaN encountered in loss at epoch {epoch + 1}")

            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            running_loss += loss.item()

            # Calculate training metrics
            mae, mse, rmse = calculate_metrics(outputs, targets)
            train_mae += mae
            train_mse += mse
            train_rmse += rmse

        # Step the learning rate scheduler
        scheduler.step()

        # Compute and store average training loss and metrics
        train_loss_avg = running_loss / len(train_loader)
        train_losses.append(train_loss_avg)

        train_mae_avg = train_mae / len(train_loader)
        train_mse_avg = train_mse / len(train_loader)
        train_rmse_avg = train_rmse / len(train_loader)

        train_mae_list.append(train_mae_avg)
        train_mse_list.append(train_mse_avg)
        train_rmse_list.append(train_rmse_avg)

        # Print training loss for each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {train_loss_avg:.4f}')

        # Evaluate the model on the test set
        model.eval()
        test_loss = 0.0
        test_mae = 0.0
        test_mse = 0.0
        test_rmse = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                # Calculate testing metrics
                mae, mse, rmse = calculate_metrics(outputs, targets)
                test_mae += mae
                test_mse += mse
                test_rmse += rmse

        # Compute and store average testing loss and metrics
        test_loss_avg = test_loss / len(test_loader)
        test_losses.append(test_loss_avg)

        test_mae_avg = test_mae / len(test_loader)
        test_mse_avg = test_mse / len(test_loader)
        test_rmse_avg = test_rmse / len(test_loader)

        test_mae_list.append(test_mae_avg)
        test_mse_list.append(test_mse_avg)
        test_rmse_list.append(test_rmse_avg)

    # Plot training and testing loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Testing Loss')
    plt.savefig('training_testing_loss.png')  # Save the plot as an image
    plt.show()

    # Summarize final evaluation metrics
    metrics = {
        'train_mae_avg': train_mae_list[-1],
        'train_mse_avg': train_mse_list[-1],
        'train_rmse_avg': train_rmse_list[-1],
        'test_mae_avg': test_mae_list[-1],
        'test_mse_avg': test_mse_list[-1],
        'test_rmse_avg': test_rmse_list[-1]
    }

    # Save the metrics as a CSV file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('evaluation_metrics.csv', index=False)

    print("Final evaluation metrics:", metrics)
    return metrics



if __name__ == '__main__':
    # Load data
    train_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_train.csv'
    test_file_path = 'D:\\Users\\allmi\\PycharmProjects\\IOTE\\Dataset\\FLIR_groups1and2_test.csv'
    df_train = load_data(train_file_path)
    df_test = load_data(test_file_path)
    df_train_x, df_train_y = split_features_labels(df_train)
    df_test_x, df_test_y = split_features_labels(df_test)

    # Define the model
    model = MLP(input_size=df_train_x.shape[1])
    epochs = 700
    batch_size = 16
    # Define criterion and hyperparameters for tuning
    criterion = nn.MSELoss()
    hyperparams = {
        'Adam': {'lr': 0.01},
        'SGD': {'lr': 0.001, 'momentum': 0.9}
    }

    # Perform hyperparameter tuning using validation
    best_params, best_metrics = hyperparameter_tuning(df_train_x, df_train_y, criterion=criterion, hyperparams=hyperparams,
                                                      num_epochs=epochs, val_split=0.2, batch_size=batch_size)

    # Perform final evaluation on the test set using the best hyperparameters
    evaluation_metrics = evaluation(model, best_params, criterion, df_train_x, df_train_y, df_test_x, df_test_y,
                                    num_epochs=epochs, batch_size=batch_size)

    # Print the final evaluation metrics
    print("Evaluation complete. Final metrics on the test set:")
    print(evaluation_metrics)