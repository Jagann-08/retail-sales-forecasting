import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("dataset/cleaned_retail_sales.csv")

df["Date"] = pd.to_datetime(df["Date"])

os.makedirs("eda_results", exist_ok=True)

print("===== DATASET SUMMARY =====")
print(df.describe())

print("\n===== SALES SUMMARY =====")
print("Total Sales:", round(df["Sales"].sum(), 2))
print("Average Sales:", round(df["Sales"].mean(), 2))
print("Minimum Sales:", round(df["Sales"].min(), 2))
print("Maximum Sales:", round(df["Sales"].max(), 2))

print("\n===== CATEGORY SALES =====")
print(
    df.groupby("Product_Category")["Sales"]
    .mean()
    .sort_values(ascending=False)
)

print("\n===== PROMOTION ANALYSIS =====")
print(
    df.groupby("Promotion")["Sales"]
    .mean()
)

print("\n===== HOLIDAY ANALYSIS =====")
print(
    df.groupby("Holiday")["Sales"]
    .mean()
)

# 1. Weekly Sales Trend
weekly_sales = (
    df.groupby("Date")["Sales"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(12, 6))
plt.plot(weekly_sales["Date"], weekly_sales["Sales"])
plt.title("Weekly Retail Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_results/1_weekly_sales_trend.png")
plt.close()

# 2. Sales by Product Category
category_sales = (
    df.groupby("Product_Category")["Sales"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
category_sales.plot(kind="bar")
plt.title("Average Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Average Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_results/2_category_sales.png")
plt.close()

# 3. Promotion vs Sales
promotion_sales = df.groupby("Promotion")["Sales"].mean()

plt.figure(figsize=(8, 6))
promotion_sales.plot(kind="bar")
plt.title("Average Sales: Promotion vs No Promotion")
plt.xlabel("Promotion (0 = No, 1 = Yes)")
plt.ylabel("Average Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_results/3_promotion_sales.png")
plt.close()

# 4. Holiday vs Sales
holiday_sales = df.groupby("Holiday")["Sales"].mean()

plt.figure(figsize=(8, 6))
holiday_sales.plot(kind="bar")
plt.title("Average Sales: Holiday vs Non-Holiday")
plt.xlabel("Holiday (0 = No, 1 = Yes)")
plt.ylabel("Average Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_results/4_holiday_sales.png")
plt.close()

# 5. Rainfall vs Sales
plt.figure(figsize=(10, 6))
plt.scatter(df["Rainfall"], df["Sales"], alpha=0.5)
plt.title("Rainfall vs Sales")
plt.xlabel("Rainfall")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("eda_results/5_rainfall_sales.png")
plt.close()

# 6. Customers vs Sales
plt.figure(figsize=(10, 6))
plt.scatter(df["Customers"], df["Sales"], alpha=0.5)
plt.title("Customers vs Sales")
plt.xlabel("Customers")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("eda_results/6_customers_sales.png")
plt.close()

print("\nEDA Completed Successfully!")

print("\nGraphs saved in:")
print("eda_results/")