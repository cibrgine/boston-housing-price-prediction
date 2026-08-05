import pandas as pd
import matplotlib.pyplot as plt
from model_uni import run_model_uni
from model_multi import run_model_multi
from model_Mix_columns import run_model_mix
from model_ridge import run_model_ridge

def compare_all_models(data_path='cleaned_house.csv'):
    df = pd.read_csv(data_path)
    
    results = {
        'Univariate (RM)': run_model_uni(df),
        'Multivariate': run_model_multi(df),
        'Mixed Columns (CRIM * PTRATIO)': run_model_mix(df),
        'RidgeCV': run_model_ridge(df)
    }
    
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.to_string())

    return comparison_df

if __name__ == "__main__":
    compare_all_models()
