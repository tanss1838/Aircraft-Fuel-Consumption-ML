# ✈️ Flight Fuel Consumption Predictor

A beginner machine learning project that predicts per-flight fuel consumption using Linear Regression, built on real-world aviation data.

---

## 📌 Project Overview

This project uses Linear Regression to predict how much fuel an aircraft burns per flight, based on route and aircraft characteristics. It is part of a series of aerospace-themed ML projects for beginners.

---

## 🎯 Motivation

Fuel is one of the largest costs in aviation and a major contributor to CO2 emissions. Being able to predict fuel consumption from basic flight parameters has real-world applications in:
- Flight planning and cost estimation
- Emissions reporting
- Fleet efficiency analysis

---

## 📂 Dataset

The dataset contains real-world aviation route data including aircraft type, route distance, number of seats, and fuel burn figures.

**Key columns used:**
- `distance_km` — great circle distance between airports
- `seats` — number of seats (proxy for aircraft size)
- `acft_class` — aircraft category (Narrow Body, Wide Body, Regional Jet, etc.)
- `domestic` — whether the flight is domestic (1) or international (0)
- `fuel_burn` — total fuel burned across all flights on a route
- `n_flights` — number of flights operated on that route

---

## 🔧 Methodology

### 1. Target Variable Engineering
- Created `fuel_per_flight = fuel_burn / n_flights` to get per-flight fuel burn instead of total route fuel
- Applied `np.log1p()` transformation to handle right-skewed distribution

### 2. Data Cleaning
- Removed rows with negative fuel values (data entry errors)
- Removed rows where departure and arrival airport were the same (dummy/placeholder rows)
- Dropped rows where `fuel_per_flight` was zero or near-zero

### 3. Feature Engineering
- Applied one-hot encoding to `acft_class` using `pd.get_dummies()`
- Selected 4 meaningful features: `distance_km`, `seats`, `domestic`, and aircraft class dummies

### 4. Model Training
- Split data 80/20 into training and test sets using `train_test_split`
- Trained a `LinearRegression` model from scikit-learn

### 5. Evaluation
- Evaluated using R² Score and RMSE
- Visualized predictions vs actuals and residual plot

---

## 📊 Results

| Metric | Value |
|--------|-------|
| R² Score | 0.896 |
| RMSE (log scale) | 0.452 |
| RMSE (actual scale) | ~1.57x average error |

The model explains **89.6%** of the variance in per-flight fuel consumption.

---

## 📈 Key Visualizations

- Histogram of log-transformed fuel per flight
- Scatter plot: Actual vs Predicted fuel burn
- Residual plot showing model error distribution

---

## ⚠️ Limitations

- The residual plot shows heteroscedasticity — errors are not fully random, suggesting a non-linear relationship
- The model overestimates fuel burn for short/small flights
- Missing features like load factor, wind conditions, and engine type would improve accuracy

---

## 🚀 Future Improvements

- Log-transform `distance_km` as well to better capture its curved relationship with fuel burn
- Add `departure_continent` as a geographic feature
- Try more advanced models like Random Forest or Gradient Boosting
- Build a simple prediction function that takes flight inputs and returns estimated fuel in kg

---

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 👤 Author

Built as a beginner ML project exploring aerospace data with Linear Regression.
