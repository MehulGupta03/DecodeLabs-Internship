import pandas as pd
# import matplotlib.pyplot as plt

df = pd.read_excel("Online-Store-Orders.xlsx")

# task 1
df.info()
print("First 5 rows:")
print(df.head())

print("\nShape:", df.shape)

print("\nSummary:")
print(df.describe())