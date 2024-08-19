# Infrared Oral Temperature Estimation

## Overview
This project aims to predict oral temperature using infrared thermography data through various machine learning models. The dataset includes 1,020 samples with 33 features, comprising demographic and environmental variables. The models used include Lasso, Ridge, K-Nearest Neighbors, Support Vector Regression, Bayesian Ridge, Random Forest, RBF Network, and Multi-Layer Perceptron (MLP). After fine-tuning hyperparameters, models were evaluated based on Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

## Project Structure

```bash
├── Dataset
│   ├── FLIR_groups1and2_train.csv     # Training dataset
│   ├── FLIR_groups1and2_test.csv      # Testing dataset
│
├── models
│   ├── 1NN.py                         # 1-Nearest Neighbor regression model
│   ├── bayesian_ridge.py              # Bayesian Ridge regression model
│   ├── knn.py                         # K-Nearest Neighbors regression model
│   ├── lasso.py                       # Lasso regression model
│   ├── linear_regression.py           # Linear regression model
│   ├── mlp.py                         # Multi-Layer Perceptron neural network model
│   ├── random_forest.py               # Random Forest regression model
│   ├── rbf_network.py                 # Radial Basis Function network model
│   ├── ridge.py                       # Ridge regression model
│   ├── svr.py                         # Support Vector Regression model
│   └── trivial_sys.py                 # Trivial baseline system using mean prediction
│
├── results
│   ├── 1NN                            # Results from 1NN model
│   ├── Bayes                          # Results from Bayesian Ridge model
│   ├── KNN                            # Results from KNN model
│   ├── Lasso                          # Results from Lasso model
│   ├── Linear_Regression              # Results from Linear Regression model
│   ├── MLP                            # Results from MLP model
│   ├── RandomForest                   # Results from Random Forest model
│   ├── RBF_Network                    # Results from RBF Network model
│   ├── Ridge                          # Results from Ridge model
│   ├── SVR                            # Results from SVR model
│   └── Trivial_System                 # Results from trivial baseline system
│
└── src
    ├── data_preprocess.py             # Data preprocessing functions
    ├── feature_engineering.py         # Feature selection and dimensionality reduction
    ├── model_evaluation.py            # Model evaluation and cross-validation
    └── visulization.py                # Visualization of model performance


## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- torch (for MLP)
- matplotlib

