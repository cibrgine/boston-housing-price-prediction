import pandas as pd

df = pd.read_csv("HousingData.csv")
df_clean = df.dropna().reset_index(drop=True)
df_clean.to_csv("cleaned_house.csv", index=False)