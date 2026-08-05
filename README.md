#  Boston House Price Prediction: Linear Regression Comparison

A machine learning project designed to benchmark and evaluate **4 different Linear Regression approaches** on the Boston Housing dataset. The core objective is to iteratively test feature selection, feature engineering, and regularization techniques to determine which model achieves optimal predictive accuracy.

---

##  Project Goal

The primary goal of this repository is to systematically compare **four Linear Regression variations** using 5-fold cross-validation on key metrics (**MSE**, **RMSE**, **MAE**, and **$R^2$**) to evaluate performance improvements across model iterations and identify the best overall model for housing price prediction.

---

## 📁 Repository Structure

```
├── HousingData.csv         # Raw Boston Housing dataset
├── cleaned_house.csv       # Preprocessed dataset (null values removed)
├── data_prep.py            # Data cleaning and preprocessing script
├── model_uni.py            # Model 1: Univariate Linear Regression (RM feature)
├── model_multi.py          # Model 2: Multivariate Linear Regression (All features)
├── model_Mix_columns.py    # Model 3: Multivariate + Interaction Feature (CRIM * PTRATIO)
├── model_ridge.py          # Model 4: Ridge Regression with Hyperparameter Tuning (RidgeCV)
├── MSE_Compare.py          # Master comparison script aggregating fold metrics
├── images/                 # Model code and benchmark output screenshots
└── requirements.txt        # Python dependency requirements
```

---

## 🧹 Data Preparation (`data_prep.py`)

### How Data Preparation Works
The dataset contains information collected by the U.S. Census Bureau concerning housing in the area of Boston, MA. The target variable is `MEDV` (Median value of owner-occupied homes in $1000's).

1. **Loading Raw Data**: Reads `HousingData.csv` into a Pandas DataFrame.
2. **Missing Value Handling**: Drops missing values using `.dropna()`.
3. **Index Resetting**: Resets row indexing (`reset_index(drop=True)`).
4. **Export**: Saves the cleaned data to `cleaned_house.csv` for standardized consumption across all model scripts.

* **Input**: `HousingData.csv`
* **Output**: `cleaned_house.csv`
* **Core Functions**: `pd.read_csv()`, `df.dropna()`, `df.reset_index()`, `df.to_csv()`

---

## 🔬 Model Specifications & Detailed File Breakdown

Every model function accepts a DataFrame and cross-validation split count, standardized through a `scikit-learn` `Pipeline` consisting of a `StandardScaler` feature transformer and a regression estimator.

### 1️⃣ Model 1: Univariate Linear Regression (`model_uni.py`)
* **Purpose**: Establish a baseline using only the single most correlated feature (`RM` - average number of rooms per dwelling).
* **Inputs**: 
  * `df` (*pd.DataFrame*): Cleaned housing DataFrame.
  * `n_splits` (*int*, default=5): Number of K-Fold CV splits.
* **Feature Matrix ($X$) & Target ($y$)**:
  * $X$: `df[['RM']]` (1 predictor)
  * $y$: `df['MEDV']`
* **Functions & Tools Used**:
  * `KFold(n_splits=5, shuffle=True, random_state=42)`: Sets up reproducible cross-validation.
  * `Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])`: Standardizes features to zero mean/unit variance and fits Ordinary Least Squares regression.
  * `cross_validate()`: Evaluates performance across the 5 validation folds.
* **Outputs**: A dictionary containing averaged metrics across folds:
  ```python
  {'MSE': float, 'RMSE': float, 'MAE': float, 'R2': float}
  ```

#### Code Screenshot:
![Univariate Model Code](images/model_uni.png)

---

### 2️⃣ Model 2: Multivariate Linear Regression (`model_multi.py`)
* **Purpose**: Expand input features from a single predictor to all available numerical predictors.
* **Inputs**:
  * `df` (*pd.DataFrame*): Cleaned housing DataFrame.
  * `n_splits` (*int*, default=5): Number of K-Fold CV splits.
* **Feature Matrix ($X$) & Target ($y$)**:
  * $X$: `df.drop(columns=['MEDV'])` (All 13 features: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT)
  * $y$: `df['MEDV']`
* **Functions & Tools Used**: `KFold`, `StandardScaler`, `LinearRegression`, `Pipeline`, `cross_validate`.
* **Outputs**: Dictionary with averaged metrics (`MSE`, `RMSE`, `MAE`, `R2`).

#### Code Screenshot:
![Multivariate Model Code](images/model_multi.png)

---

### 3️⃣ Model 3: Interaction Terms (`model_Mix_columns.py`)
* **Purpose**: Test manual feature engineering by creating an interaction term combining crime rate and pupil-teacher ratio.
* **Inputs**:
  * `df` (*pd.DataFrame*): Cleaned housing DataFrame.
  * `n_splits` (*int*, default=5): Number of K-Fold CV splits.
* **Feature Matrix ($X$) & Target ($y$)**:
  * $X$: `df.drop(columns=['MEDV'])` extended with `X['CRIM_PTRATIO'] = X['CRIM'] * X['PTRATIO']` (14 predictors).
  * $y$: `df['MEDV']`
* **Functions & Tools Used**: `KFold`, `StandardScaler`, `LinearRegression`, `Pipeline`, `cross_validate`.
* **Outputs**: Dictionary with averaged metrics (`MSE`, `RMSE`, `MAE`, `R2`).

#### Code Screenshot:
![Mixed Columns Model Code](images/model_mix.png)

---

### 4️⃣ Model 4: Ridge Regression CV (`model_ridge.py`)
* **Purpose**: Prevent potential multicollinearity and overfitting by adding $L_2$ regularization with automatic hyperparameter selection ($\alpha$).
* **Inputs**:
  * `df` (*pd.DataFrame*): Cleaned housing DataFrame.
  * `n_splits` (*int*, default=5): Number of K-Fold CV splits.
  * `alphas` (*array-like*, default=`np.logspace(-3, 3, 100)`): Range of 100 candidate regularization parameters from $10^{-3}$ to $10^3$.
* **Feature Matrix ($X$) & Target ($y$)**:
  * $X$: `df.drop(columns=['MEDV'])` (13 features).
  * $y$: `df['MEDV']`
* **Functions & Tools Used**:
  * `RidgeCV(alphas=alphas)`: Fits Ridge Regression model while performing internal cross-validation to pick the optimal regularization weight $\alpha$.
  * `Pipeline`, `StandardScaler`, `KFold`, `cross_validate`.
* **Outputs**: Dictionary with averaged metrics (`MSE`, `RMSE`, `MAE`, `R2`).

#### Code Screenshot:
![RidgeCV Model Code](images/model_ridge.png)

---

### 5️⃣ Master Comparison Script (`MSE_Compare.py`)
* **Purpose**: Orchestrate model executions, gather cross-validation performance, and format benchmark results.
* **Inputs**: Path to cleaned dataset (defaulting to `cleaned_house.csv`).
* **Workflow**:
  1. Calls `run_model_uni(df)`
  2. Calls `run_model_multi(df)`
  3. Calls `run_model_mix(df)`
  4. Calls `run_model_ridge(df)`
  5. Aggregates returned metrics into a comparative Pandas DataFrame.
* **Outputs**: Prints formatted markdown/string comparison table and returns the aggregated DataFrame.

#### Code Screenshot:
![MSE Compare Script Code](images/mse_compare_code.png)

---

## 📊 Benchmark Results

Running `python MSE_Compare.py` evaluates all 4 models over 5-fold cross-validation:

| Model Strategy | MSE (Mean Squared Error) ↓ | RMSE (Root MSE) ↓ | MAE (Mean Absolute Error) ↓ | $R^2$ (Variance Explained) ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **Univariate (`RM`)** | 40.3758 | 6.3542 | 4.3259 | 0.5089 |
| **Multivariate** | 21.8956 | 4.6793 | 3.2798 | 0.7330 |
| **Mixed Columns (`CRIM * PTRATIO`)** | 21.9623 | 4.6864 | 3.2675 | 0.7325 |
| **RidgeCV** | **21.8334** | **4.6726** | **3.2346** | **0.7340** |

#### Terminal Benchmark Output Screenshot:
![Benchmark Results](images/benchmark_results.png)

---

## 📈 Model Evolution & Performance Analysis

### 1. Univariate vs. Multivariate
* **Evolution**: Transitioning from a single feature (`RM`) to all 13 features.
* **Why Multivariate is Better**: 
  * The univariate model relies entirely on room count. While `RM` is a strong price signal ($R^2 \approx 50.9\%$), single-variable regression misses critical socioeconomic factors (e.g., `LSTAT` - lower status population percentage, `PTRATIO` - pupil-teacher ratio, and `TAX` - property tax rate).
  * Adding all features cuts MSE nearly in half (from **40.38** to **21.90**) and boosts explained variance ($R^2$) from **50.9%** to **73.3%**.

### 2. Multivariate vs. Mixed Columns (Interaction Term)
* **Evolution**: Adding an engineered feature (`CRIM * PTRATIO`).
* **Analysis**:
  * MSE slightly increased from **21.8956** to **21.9623** (though MAE showed a minor improvement from 3.2798 to 3.2675).
  * Naive feature interaction does not guarantee better generalization. The raw features `CRIM` and `PTRATIO` already contribute linearly; creating a multiplicative product added slight redundant variance/collinearity without adding new signal.

### 3. Multivariate vs. RidgeCV (Why RidgeCV is the Best Model)
* **Evolution**: Applying $L_2$ Tikhonov Regularization ($RidgeCV$) to penalty-constrain regression weights.
* **Why RidgeCV is the Best Overall Model**:
  1. **Optimal Performance**: Achieves the **lowest MSE (21.8334)**, **lowest RMSE (4.6726)**, **lowest MAE (3.2346)**, and **highest $R^2$ (0.7340 / 73.4%)**.
  2. **Multicollinearity Suppression**: Features in the Boston Housing dataset (such as `TAX` and `RAD`, or `NOX` and `DIS`) exhibit strong collinearities. Standard Ordinary Least Squares (OLS) produces inflated regression coefficients when features are correlated.
  3. **Variance Reduction**: $L_2$ regularization penalizes large coefficient magnitudes ($\lambda \sum \beta_i^2$), reducing variance without adding significant bias.
  4. **Automated Alpha Search**: `RidgeCV` evaluates 100 log-spaced $\alpha$ parameters internally to select the optimal regularization strength for maximum out-of-fold generalization.

---

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess Data** (Optional if `cleaned_house.csv` already exists):
   ```bash
   python data_prep.py
   ```

3. **Run Model Benchmark & Compare Results**:
   ```bash
   python MSE_Compare.py
   ```
