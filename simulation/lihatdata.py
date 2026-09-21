import pandas as pd

df = pd.read_csv("simulation/customers.csv")

print(df.shape)
print(df.dtypes)
print(df.head())

print(
    df.isna()
      .mean()
      .sort_values(ascending=False)
      .head()
)