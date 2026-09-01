# Retail Sales Forecasting Using Machine Learning and Tableau

## Project Overview

This project focuses on forecasting retail store sales using historical sales data along with promotions, holidays, customer count, weather conditions, discounts, and stock availability.

A Random Forest Regression model is used to predict future sales. Tableau is used to visualize sales trends, category performance, promotion and holiday impacts, and future sales forecasts.

## Objectives

- Analyse historical retail sales data
- Perform data preprocessing
- Perform exploratory data analysis
- Analyse promotion and holiday impacts
- Develop a machine learning model
- Forecast sales for the next four weeks
- Create a Tableau dashboard

## Dataset

The dataset contains 1,000 retail sales records and 17 attributes after feature engineering.

Important features include:

- Date
- Store ID
- Product Category
- Sales
- Promotion
- Holiday
- Customers
- Temperature
- Rainfall
- Discount Percentage
- Stock Available
- Previous Week Sales
- Moving Average

## Machine Learning Model

Random Forest Regression was used for sales prediction.

### Model Performance

- MAE: 1012.02
- RMSE: 1248.43
- R² Score: 0.8874

## Forecast

The trained model was used to forecast retail sales for the next four weeks.

## Visualization

Tableau was used to create an interactive dashboard containing:

- Weekly Sales Trend
- Category-wise Sales
- Promotion Impact
- Holiday Impact
- Four-Week Sales Forecast

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Tableau
- CSV

## Project Structure

```text
retail-sales-forecasting/
├── Python source files
├── dataset/
├── visualizations/
├── tableau/
└── README.md
