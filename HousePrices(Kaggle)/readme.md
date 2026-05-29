# House Prices: Advanced Regression Techniques 🏠

## 📌 Project Overview
The primary goal of this project is to predict the final sale price of residential homes in Ames, Iowa, based on 79 explanatory variables (both numerical and categorical). This project was developed as part of a Kaggle competition. The main focus was placed on building robust data workflows (Pipelines) and applying regularization techniques to mitigate overfitting.

## 🛠 Tech Stack
* **Language:** Python
* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`
* **Environment Management:** `uv`

## 🧠 Methodology & Architecture

### 1. Data Preprocessing
A dedicated `ColumnTransformer` was constructed to encapsulate operations and prevent data leakage between the train and test sets:
* **Numerical Features:** Missing values were imputed using the median (`SimpleImputer`), followed by scaling using `StandardScaler`.
* **Categorical Features:** Missing values were filled with the most frequent value, and then encoded using `OneHotEncoder` (ignoring unknown categories).

### 2. Modeling & Regularization
To handle multicollinearity and reduce noise, three regularized linear regression models were evaluated:
* **Ridge Regression (L2)**
* **Lasso Regression (L1)** - This algorithm additionally performed automatic feature selection by shrinking the weights of the least important features to zero.
* **ElasticNet (L1 + L2)**

All estimators were directly integrated into the `Pipeline` object.

### 3. Hyperparameter Tuning
`GridSearchCV` with 5-fold Cross-Validation was utilized to systematically optimize the regularization strength (`alpha`) and the ratio (`l1_ratio` for ElasticNet).

## 📊 Results
* **Final Selected Model:** Lasso Regression (optimal `alpha` = 190)
* **Kaggle Evaluation Score (RMSLE - Root Mean Squared Logarithmic Error):** 0.13893

## 🚀 Local Setup
This project uses the modern `uv` package manager. To replicate the environment and run the notebook locally, follow these steps:

1. Clone the repository to your local machine.
2. Set up the virtual environment and install dependencies:
```bash
uv venv
uv pip install -r requirements.txt