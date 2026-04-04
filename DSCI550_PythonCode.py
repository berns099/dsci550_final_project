# =====================================
# DSCI 550 Project - Professor Seon Kim
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------
# 1. LOAD DATA and SUMMARY
# -------------------------
df = pd.read_csv(r"C:\Users\kyung\Desktop\Caleb Folder\USC Work\DSCI 550 - Seon Kim\Project\Version 3 Idea - Movie\tmdb_5000_movies.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())


# ------------------
# 2. BASIC CLEANING
# ------------------

# budget: Keep only rows with realistic budget and valid revenue (0 < budget > 10000)
df = df[(df["budget"] > 10000) & (df["revenue"] > 0)]

# release_date: Convert release_date to datetime and extract release_year
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = df["release_date"].dt.year

# Get features we can use for predicting revenue
features = ["budget", "runtime", "popularity", "vote_average", "vote_count", "release_year"]
target = "revenue"

df = df[features + [target]].dropna()

# To reduce skewness on revenue and handle potential 0 value, "log1p" is Applied
# To get the real dollar amount back later, "expm1" must be applied.
df["log_revenue"] = np.log1p(df["revenue"])

print("\nShape after cleaning:", df.shape)
print("\nColumns after cleaning:")
print(df.columns.tolist())

print("\n-------------------------------------------------------------------------------------------------------------------")

# -----------------------------
# 3. EXPLORATORY DATA ANALYSIS
# -----------------------------

print("\nSummary statistics:")
print(df.describe())

# Figure 1: Historgram of Revenue and Log Revenue
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#   Subplot 1: Raw Revenue (Left)
axes[0].hist(df["revenue"], bins=40, color='skyblue', edgecolor='black')
axes[0].set_title("Distribution of Revenue (Skewed)")
axes[0].set_xlabel("Revenue")
axes[0].set_ylabel("Frequency")

#   Subplot 2: Log Revenue (Right)
axes[1].hist(df["log_revenue"], bins=40, color='lightgreen', edgecolor='black')
axes[1].set_title("Distribution of log(Revenue) (Normalized)")
axes[1].set_xlabel("log(Revenue)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()

# Figure 2: Correlation matrix
corr = df[features + ["revenue", "log_revenue"]].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


print("\nCorrelation with revenue:")
print(corr["revenue"].sort_values(ascending=False))

print("\nCorrelation with log_revenue:")
print(corr["log_revenue"].sort_values(ascending=False))

# # Scatter plot: budget vs revenue
# plt.figure(figsize=(6, 4))
# plt.scatter(df["budget"], df["revenue"], alpha=0.5)
# plt.title("Budget vs Revenue")
# plt.xlabel("Budget")
# plt.ylabel("Revenue")
# plt.tight_layout()
# plt.show()

# # Scatter plot: popularity vs revenue
# plt.figure(figsize=(6, 4))
# plt.scatter(df["popularity"], df["revenue"], alpha=0.5)
# plt.title("Popularity vs Revenue")
# plt.xlabel("Popularity")
# plt.ylabel("Revenue")
# plt.tight_layout()
# plt.show()

# ------------------
# 4. PREPARE X AND y
# ------------------

X = df[features]
y = df["log_revenue"]   # Use log(revenue) as target


# ----------------------
# 5. TRAIN / TEST SPLIT
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# --------------------
# 6. LINEAR REGRESSION
# --------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)

print("\n==============================")
print("Linear Regression Results")
print("==============================")
print("R^2 :", lr_r2)
print("MAE :", lr_mae)
print("MSE :", lr_mse)


# ---------------------------
# 7. DECISION TREE REGRESSION
# ---------------------------
dt_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_r2 = r2_score(y_test, dt_pred)
dt_mae = mean_absolute_error(y_test, dt_pred)
dt_mse = mean_squared_error(y_test, dt_pred)

print("\n==============================")
print("Decision Tree Regression Results")
print("==============================")
print("R^2 :", dt_r2)
print("MAE :", dt_mae)
print("MSE :", dt_mse)

# Feature importance
dt_importance = pd.DataFrame({
    "Feature": features,
    "Importance": dt_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nDecision Tree Feature Importance:")
print(dt_importance)

# -------------------
# 8. MODEL COMPARISON
# -------------------
results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree Regression"],
    "R^2": [lr_r2, dt_r2],
    "MAE": [lr_mae, dt_mae],
    "MSE": [lr_mse, dt_mse]
})

print("\n==============================")
print("Model Comparison")
print("==============================")
print(results)

# Plot R^2 comparison
plt.figure(figsize=(6, 4))
plt.bar(results["Model"], results["R^2"])
plt.title("Model Comparison: R^2")
plt.ylabel("R^2")
plt.tight_layout()
plt.show()

# ----------------------
# 9. ACTUAL VS PREDICTED
# ----------------------
# Linear Regression
plt.figure(figsize=(6, 4))
plt.scatter(y_test, lr_pred, alpha=0.5)
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual log(Revenue)")
plt.ylabel("Predicted log(Revenue)")
plt.tight_layout()

# Decision Tree
plt.figure(figsize=(6, 4))
plt.scatter(y_test, dt_pred, alpha=0.5)
plt.title("Decision Tree: Actual vs Predicted")
plt.xlabel("Actual log(Revenue)")
plt.ylabel("Predicted log(Revenue)")
plt.tight_layout()
plt.show()

# --------------------------------------
# 10. EXAMPLE PREDICTION FOR A NEW MOVIE
# --------------------------------------
# Example movie features:
# budget, runtime, popularity, vote_average, vote_count, release_year
new_movie = pd.DataFrame({
    "budget": [150000000],
    "runtime": [130],
    "popularity": [45],
    "vote_average": [7.5],
    "vote_count": [1200],
    "release_year": [2020]
})

# Predict log(revenue)
lr_new_log = lr_model.predict(new_movie)[0]
dt_new_log = dt_model.predict(new_movie)[0]

# Convert back to revenue
lr_new_revenue = np.expm1(lr_new_log)
dt_new_revenue = np.expm1(dt_new_log)

print("\n==============================")
print("Prediction Example")
print("==============================")
print("Linear Regression predicted revenue: $", round(lr_new_revenue, 2))
print("Decision Tree predicted revenue:   $", round(dt_new_revenue, 2))

# # ------------
# # SAVE RESULTS
# # ------------
# results.to_csv("tmdb_regression_results.csv", index=False)
# lr_coef.to_csv("linear_regression_coefficients.csv", index=False)
# dt_importance.to_csv("decision_tree_feature_importance.csv", index=False)

# print("\nSaved files:")
# print("- tmdb_regression_results.csv")
# print("- linear_regression_coefficients.csv")
# print("- decision_tree_feature_importance.csv")