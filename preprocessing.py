import pandas as pd

df = pd.read_csv("dataset/retail_sales.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())
df["Date"] = pd.to_datetime(df["Date"])

print("\nDate Data Type After Conversion:")
print(df["Date"].dtype)
df["Date"] = pd.to_datetime(df["Date"])

print("\nDate Data Type After Conversion:")
print(df["Date"].dtype)
print("\nInvalid Value Check:")

print("Negative Sales:", (df["Sales"] < 0).sum())
print("Negative Customers:", (df["Customers"] < 0).sum())
print("Negative Rainfall:", (df["Rainfall"] < 0).sum())
print("Invalid Discount:", (
    (df["Discount_Percentage"] < 0) |
    (df["Discount_Percentage"] > 100)
).sum())

print("Invalid Promotion:", (
    ~df["Promotion"].isin([0, 1])
).sum())

print("Invalid Holiday:", (
    ~df["Holiday"].isin([0, 1])
).sum())
print("\nOutlier Detection using IQR:")

outlier_columns = [
    "Sales",
    "Customers",
    "Temperature",
    "Rainfall",
    "Discount_Percentage",
    "Stock_Available"
]

for col in outlier_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ]

    print(f"{col}: {len(outliers)} outliers")
print("\nOutlier Value Inspection:")

for col in outlier_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ][col]

    if len(outliers) > 0:
        print(f"\n{col}:")
        print(outliers.to_list())
# Sort data
df = df.sort_values(["Store_ID", "Date"])

# Create time-based features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
df["Quarter"] = df["Date"].dt.quarter

# Previous sales
df["Previous_Week_Sales"] = (
    df.groupby("Store_ID")["Sales"].shift(1)
)

# Fill first available previous-sales value
df["Previous_Week_Sales"] = df["Previous_Week_Sales"].fillna(
    df["Sales"].median()
)

# 4-week moving average
df["Moving_Average_4_Weeks"] = (
    df.groupby("Store_ID")["Sales"]
    .transform(lambda x: x.rolling(4, min_periods=1).mean())
)

# Save cleaned dataset
df.to_csv(
    "dataset/cleaned_retail_sales.csv",
    index=False
)

print("\nPreprocessing Completed Successfully!")

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Columns:")
print(df.columns.tolist())

print("\nCleaned Dataset Saved As:")
print("dataset/cleaned_retail_sales.csv")