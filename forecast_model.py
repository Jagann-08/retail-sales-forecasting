import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_retail_sales.csv")

df["Date"] = pd.to_datetime(df["Date"])

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# Features
features = [
    "Promotion",
    "Holiday",
    "Customers",
    "Temperature",
    "Rainfall",
    "Discount_Percentage",
    "Stock_Available",
    "Year",
    "Month",
    "Week",
    "Quarter",
    "Previous_Week_Sales",
    "Moving_Average_4_Weeks"
]

X = df[features]
y = df["Sales"]

# Time-based train/test split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction on test data
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("===== MODEL PERFORMANCE =====")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))

# Actual vs predicted
results = df.iloc[split:].copy()

results["Predicted_Sales"] = y_pred

results[
    [
        "Date",
        "Sales",
        "Predicted_Sales",
        "Promotion",
        "Holiday"
    ]
].to_csv(
    "dataset/actual_vs_predicted.csv",
    index=False
)

print("\nActual vs Predicted file created!")

# Feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n===== FEATURE IMPORTANCE =====")
print(importance)

importance.to_csv(
    "dataset/feature_importance.csv",
    index=False
)

print("\nFeature importance file created!")
# ===== NEXT 4 WEEKS SALES FORECAST =====

last_date = df["Date"].max()

last_row = df.iloc[-1].copy()

future_predictions = []

for i in range(1, 5):

    future_date = last_date + pd.Timedelta(weeks=i)

    future_data = {
        "Promotion": int(last_row["Promotion"]),
        "Holiday": int(last_row["Holiday"]),
        "Customers": float(last_row["Customers"]),
        "Temperature": float(last_row["Temperature"]),
        "Rainfall": float(last_row["Rainfall"]),
        "Discount_Percentage": float(last_row["Discount_Percentage"]),
        "Stock_Available": float(last_row["Stock_Available"]),
        "Year": future_date.year,
        "Month": future_date.month,
        "Week": int(future_date.isocalendar().week),
        "Quarter": future_date.quarter,
        "Previous_Week_Sales": float(last_row["Sales"]),
        "Moving_Average_4_Weeks": float(
            df["Sales"].tail(4).mean()
        )
    }

    future_df = pd.DataFrame([future_data])

    prediction = model.predict(
        future_df[features]
    )[0]

    future_predictions.append({
        "Forecast_Date": future_date,
        "Predicted_Sales": round(prediction, 2)
    })

    last_row["Sales"] = prediction

forecast_df = pd.DataFrame(future_predictions)

forecast_df.to_csv(
    "dataset/next_4_weeks_forecast.csv",
    index=False
)

print("\n===== NEXT 4 WEEKS SALES FORECAST =====")
print(forecast_df)

print("\nForecast file created:")
print("dataset/next_4_weeks_forecast.csv")