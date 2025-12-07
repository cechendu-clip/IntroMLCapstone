# IntroMLCapstone – House Price Prediction

This repository contains my Intro to Machine Learning capstone project for predicting house sale prices using the Kaggle **“House Prices – Advanced Regression Techniques”** dataset.

The project compares three models learned in class (Linear Regression, Ridge Regression, and an MLP Regressor) with two literature-based ensemble models (Random Forest and Gradient Boosting), following a consistent preprocessing and evaluation pipeline.

---

## Repository Structure

- `linear_regression.ipynb`  
  Implements **Multiple Linear Regression** as a baseline model.  
  Includes:
  - Loading and preprocessing the Kaggle housing data  
  - Log-transforming the target (`SalePrice`)  
  - Fitting the linear regression model  
  - Computing metrics (MSE, MAE, RMSE, R²)  
  - Residual and actual-vs-predicted plots

- `ridge_regressor.ipynb`  
  Implements **Ridge Regression** (L2-regularized linear model).  
  Includes:
  - Same preprocessing pipeline as Linear Regression  
  - Tuning the regularization strength (alpha)  
  - Evaluating performance and comparing to the OLS baseline  
  - Visual diagnostics of residuals and prediction errors

- `mlp_regression.ipynb`  
  Implements a **Multi-Layer Perceptron (MLP) Regressor** as a neural network model.  
  Includes:
  - Preprocessing with scaled numerical features and one-hot encoded categoricals  
  - Basic hyperparameter choices (hidden layers, activation, etc.)  
  - Training/validation split and performance metrics  
  - Discussion of stability and sensitivity on tabular data

- `random_forest_literature.ipynb`  
  Re-implements a **Random Forest Regressor** based on a literature source that uses the Kaggle housing dataset.  
  Includes:
  - Similar preprocessing pipeline (imputation, encoding, scaling where needed)  
  - Random Forest training with selected hyperparameters  
  - Error metrics and comparisons to the linear and MLP models  
  - Visual comparisons (residual plots, actual vs predicted)

- `gradient_boosting_literature.ipynb`  
  Re-implements a **Gradient Boosting Regressor** from a literature paper on house price prediction.  
  Includes:
  - Preprocessing consistent with the other models  
  - Gradient Boosting training, possibly with robust loss (e.g., L1)  
  - Performance metrics and plots  
  - Comparison against Random Forest and the linear baselines

- `train.csv`  
  Training split of the Kaggle House Prices dataset used to fit and validate the models.

- `test.csv`  
  Test portion of the Kaggle dataset. It is included for completeness but is **not used** in this project’s internal training and evaluation pipeline, since it does not contain the target `SalePrice` column.

- `sample_submission.csv`  
  Template file from the Kaggle competition (ID + predicted `SalePrice`). It can be used to format predictions for a Kaggle submission, but it is **not required** for the main analysis in this project.

- `requirements.txt`

  Specifies the exact Python package versions needed to run all notebooks in this project.

---

## How to Run

1. Clone this repository:
   ```bash
   git clone  https://github.com/cechendu-clip/IntroMLCapstone.git
   cd IntroMLCapstone
2. Install required Python packages
   ```bash
   pip install -r requirements.txt
3. Open the notebooks in Jupyter or VS Code and run them cell-by-cell:
   

  
