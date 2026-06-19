# task 2

import pandas as pd
df = pd.read_excel("Online-Store-Orders.xlsx")

print(df.isnull().sum())
df["CouponCode"] = df["CouponCode"].fillna("No Coupon")
print(df.isnull().sum())
print("duplicates : ", df.duplicated().sum())
df = df.drop_duplicates()
df["Date"] = pd.to_datetime(df["Date"])
df["Product"] = df["Product"].str.strip()
df["PaymentMethod"] = df["PaymentMethod"].str.strip()
df["OrderStatus"] = df["OrderStatus"].str.strip()
df["PaymentMethod"] = df["PaymentMethod"].str.lower()
df["OrderStatus"] = df["OrderStatus"].str.lower()
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"])
df["TotalPrice"] = pd.to_numeric(df["TotalPrice"])