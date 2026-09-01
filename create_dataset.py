import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

dates = pd.date_range(
    start="2024-01-07",
    periods=n,
    freq="W"
)

store_ids = np.random.choice(
    ["S001", "S002", "S003", "S004", "S005"],
    n
)

categories = np.random.choice(
    ["Grocery", "Electronics", "Clothing", "Home", "Beauty"],
    n
)

promotion = np.random.choice(
    [0, 1],
    n,
    p=[0.65, 0.35]
)

holiday = np.random.choice(
    [0, 1],
    n,
    p=[0.85, 0.15]
)

temperature = np.random.normal(30, 4, n).round(1)
temperature = np.clip(temperature, 18, 42)

rainfall = np.random.gamma(2, 5, n).round(1)
rainfall = np.clip(rainfall, 0, 50)

discount = np.where(
    promotion == 1,
    np.random.uniform(5, 30, n),
    np.random.uniform(0, 5, n)
).round(1)

customers = (
    500
    + promotion * 180
    + holiday * 220
    - rainfall * 2
    + np.random.normal(0, 70, n)
).round().astype(int)

customers = np.clip(customers, 200, 1500)

stock_available = (
    1000
    + customers * 0.6
    + promotion * 250
    + np.random.normal(0, 100, n)
).round().astype(int)

sales = (
    5000
    + customers * 12
    + promotion * 2500
    + holiday * 3000
    + discount * 100
    - rainfall * 40
    + temperature * 20
    + np.random.normal(0, 1200, n)
)

sales = np.maximum(sales, 2000).round(2)

df = pd.DataFrame({
    "Date": dates,
    "Store_ID": store_ids,
    "Product_Category": categories,
    "Sales": sales,
    "Promotion": promotion,
    "Holiday": holiday,
    "Customers": customers,
    "Temperature": temperature,
    "Rainfall": rainfall,
    "Discount_Percentage": discount,
    "Stock_Available": stock_available
})

df = df.sort_values(["Store_ID", "Date"])

df.to_csv(
    "dataset/retail_sales.csv",
    index=False
)

print("Dataset created successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())