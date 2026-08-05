import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def run_model_multi(df, n_splits=5):
    X = df.drop(columns=['MEDV'])
    y = df['MEDV']
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    scoring = {'neg_mse': 'neg_mean_squared_error', 'neg_mae': 'neg_mean_absolute_error', 'r2': 'r2'}
    results = cross_validate(pipeline, X, y, cv=kf, scoring=scoring)
    
    mse = -np.mean(results['test_neg_mse'])
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': -np.mean(results['test_neg_mae']),
        'R2': np.mean(results['test_r2'])
    }