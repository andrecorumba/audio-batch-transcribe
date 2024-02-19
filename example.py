import pandas as pd

df = pd.read_parquet('data/validation-00000-of-00001-913508124a40cb97.parquet')

print(df.head())