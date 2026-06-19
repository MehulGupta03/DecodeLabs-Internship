import pandas as pd
df = pd.read_excel("Online-Store-Orders.xlsx")
import matplotlib.pyplot as plt


df.groupby("Product")["TotalPrice"].sum().sort_values(ascending=False).head().plot(kind='bar')
plt.title("Top Revenue Products")
plt.xlabel("Product")
plt.ylabel("Revenue")
plt.show()
