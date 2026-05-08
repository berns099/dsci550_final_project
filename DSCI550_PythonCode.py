# ==========================================================================================
# DSCI 550 Project - Hit or Flop? Predicting Movie Box Office Revenue Using Machine Learning
#
# Group Members: Teresa Lee, Bernard Yu, Kyung Je Park, Stephen Rosario
# ==========================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Regression models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Regression metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Classification models
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Classification metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# -------------------------
# 1. LOAD DATA and SUMMARY
# -------------------------
csv_path = input("Enter the path to tmdb_5000_movies.csv: ").strip().strip('"').strip("'")
df = pd.read_csv(csv_path)

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

# Subplot 1: Raw Revenue (Left)
axes[0].hist(df["revenue"], bins=40, color='skyblue', edgecolor='black')
axes[0].set_title("Distribution of Revenue (Skewed)")
axes[0].set_xlabel("Revenue")
axes[0].set_ylabel("Frequency")

# Subplot 2: Log Revenue (Right)
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

X = df[features]
y = df["log_revenue"]   # Use log(revenue) as target


# ----------------------------
# 4-a. REGRESSION: DEVELOPMENT
# ----------------------------

# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nRegression training set shape:", X_train.shape)
print("Regression test set shape:", X_test.shape)

# LINEAR REGRESSION
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

# RIDGE REGRESSION
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_test)

ridge_r2 = r2_score(y_test, ridge_pred)
ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)

print("\n==============================")
print("Ridge Regression Results")
print("==============================")
print("R^2 :", ridge_r2)
print("MAE :", ridge_mae)
print("MSE :", ridge_mse)

# DECISION TREE REGRESSION
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

# RANDOM FOREST
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)

print("\n==============================")
print("Random Forest Results")
print("==============================")
print("R^2 :", rf_r2)
print("MAE :", rf_mae)
print("MSE :", rf_mse)

rf_importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_importance)


# ---------------------------------
# 4-b. REGRESSION: MODEL COMPARISON
# ---------------------------------
results = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression", "Decision Tree Regression", "Random Forest"],
    "R^2": [lr_r2, ridge_r2, dt_r2, rf_r2],
    "MAE": [lr_mae, ridge_mae, dt_mae, rf_mae],
    "MSE": [lr_mse, ridge_mse, dt_mse, rf_mse]
})
print("\n==============================")
print("Model Comparison")
print("==============================")
print(results)

# Plot R^2 comparison
plt.figure(figsize=(8, 4))
plt.bar(results["Model"], results["R^2"], color=["skyblue", "steelblue", "lightgreen", "seagreen"])
plt.title("Model Comparison: R^2")
plt.ylabel("R^2")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

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

# ------------------------------------
# 4-c. REGRESSION: ACTUAL VS PREDICTED
# ------------------------------------
# Linear Regression
plt.figure(figsize=(6, 4))
plt.scatter(y_test, lr_pred, alpha=0.5)
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual log(Revenue)")
plt.ylabel("Predicted log(Revenue)")
plt.tight_layout()

# Ridge Regression
plt.figure(figsize=(6, 4))
plt.scatter(y_test, ridge_pred, alpha=0.5)
plt.title("Ridge Regression: Actual vs Predicted")
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

# Random Forest
plt.figure(figsize=(6, 4))
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.title("Random Forest: Actual vs Predicted")
plt.xlabel("Actual log(Revenue)")
plt.ylabel("Predicted log(Revenue)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 4-d. REGRESSION: EXAMPLE PREDICTION FOR A NEW MOVIE
# ---------------------------------------------------
# Example movie features:
# budget, runtime, popularity, vote_average, vote_count, release_year

new_movie_1 = pd.DataFrame({
    "budget": [150000000],
    "runtime": [130],
    "popularity": [200],
    "vote_average": [8.5],
    "vote_count": [4000],
    "release_year": [2020]
})
new_movie_2 = pd.DataFrame({
    "budget": [80000000],
    "runtime": [90],
    "popularity": [3],
    "vote_average": [3.5],
    "vote_count": [500],
    "release_year": [2015]
})

# Predict log(revenue) for 1st movie
lr_new_log1 = lr_model.predict(new_movie_1)[0] # Linear Regression
ridge_new_log1 = ridge_model.predict(new_movie_1)[0] # Ridge Regression
dt_new_log1 = dt_model.predict(new_movie_1)[0] # Decision Tree
rf_new_log1 = rf_model.predict(new_movie_1)[0] # Random Forest

# Predict log(revenue) for 2nd movie
lr_new_log2 = lr_model.predict(new_movie_2)[0] # Linear Regression
ridge_new_log2 = ridge_model.predict(new_movie_2)[0] # Ridge Regression
dt_new_log2 = dt_model.predict(new_movie_2)[0] # Decision Tree
rf_new_log2 = rf_model.predict(new_movie_2)[0] # Random Forest

# Convert back to revenue for 1st movie
lr_new_revenue1 = np.expm1(lr_new_log1)
ridge_new_revenue1 = np.expm1(ridge_new_log1)
dt_new_revenue1 = np.expm1(dt_new_log1)
rf_new_revenue1 = np.expm1(rf_new_log1)

# Convert back to revenue for 2nd movie
lr_new_revenue2 = np.expm1(lr_new_log2)
ridge_new_revenue2 = np.expm1(ridge_new_log2)
dt_new_revenue2 = np.expm1(dt_new_log2)
rf_new_revenue2 = np.expm1(rf_new_log2)

print("\n===================")
print("Predicted Revenue")
print("===================")
print("1st Movie")
print("Linear Regression predicted revenue: $", round(lr_new_revenue1, 2))
print("Ridge Regression predicted revenue: $", round(ridge_new_revenue1, 2))
print("Decision Tree predicted revenue:   $", round(dt_new_revenue1, 2))
print("Random Forest predicted revenue:   $", round(rf_new_revenue1, 2))
print("\n2nd Movie")
print("Linear Regression predicted revenue: $", round(lr_new_revenue2, 2))
print("Ridge Regression predicted revenue: $", round(ridge_new_revenue2, 2))
print("Decision Tree predicted revenue:   $", round(dt_new_revenue2, 2))
print("Random Forest predicted revenue:   $", round(rf_new_revenue2, 2))

# =================================
# 5-a. CLASSIFICATION: DEVELOPMENT
# =================================

# Create Classification Target:
# Hit = revenue >= 2.5 * budget
# Flop = revenue < 2.5 * budget

df["hit_flop"] = np.where(df["revenue"] >= 2.5 * df["budget"], 1, 0)

print("\n==============================")
print("Hit vs Flop Class Distribution")
print("==============================")
print(df["hit_flop"].value_counts())
print("Ratio", df["hit_flop"].value_counts(normalize=True) * 100)

X_class = df[features]
y_class = df["hit_flop"]

# TRAIN / TEST SPLIT
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)

print("\nClassification training set shape:", X_train_c.shape)
print("Classification test set shape:", X_test_c.shape)


# DECISION TREE CLASSIFIER
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)

dt_classifier.fit(X_train_c, y_train_c)
dt_class_pred = dt_classifier.predict(X_test_c)

dt_class_accuracy = accuracy_score(y_test_c, dt_class_pred)
dt_class_precision = precision_score(y_test_c, dt_class_pred)
dt_class_recall = recall_score(y_test_c, dt_class_pred)
dt_class_f1 = f1_score(y_test_c, dt_class_pred)

print("\n===================================")
print("Decision Tree Classification Results")
print("===================================")
print("Accuracy :", dt_class_accuracy)
print("Precision:", dt_class_precision)
print("Recall   :", dt_class_recall)
print("F1 Score :", dt_class_f1)


# NAIVE BAYES CLASSIFIER
nb_classifier = GaussianNB()

nb_classifier.fit(X_train_c, y_train_c)
nb_class_pred = nb_classifier.predict(X_test_c)

nb_class_accuracy = accuracy_score(y_test_c, nb_class_pred)
nb_class_precision = precision_score(y_test_c, nb_class_pred)
nb_class_recall = recall_score(y_test_c, nb_class_pred)
nb_class_f1 = f1_score(y_test_c, nb_class_pred)

print("\n==============================")
print("Naive Bayes Classification Results")
print("==============================")
print("Accuracy :", nb_class_accuracy)
print("Precision:", nb_class_precision)
print("Recall   :", nb_class_recall)
print("F1 Score :", nb_class_f1)

# -------------------------------------
# 5-b. CLASSIFICATION: MODEL COMPARISON
# -------------------------------------

classification_results = pd.DataFrame({
    "Model": ["Decision Tree Classifier", "Naive Bayes"],
    "Accuracy": [dt_class_accuracy, nb_class_accuracy],
    "Precision": [dt_class_precision, nb_class_precision],
    "Recall": [dt_class_recall, nb_class_recall],
    "F1 Score": [dt_class_f1, nb_class_f1]
})

print("\n==============================")
print("Classification Model Comparison")
print("==============================")
print(classification_results)

# Plot classification metrics
plt.figure(figsize=(8, 4))
plt.bar(classification_results["Model"], classification_results["Accuracy"])
plt.title("Classification Model Comparison: Accuracy")
plt.ylabel("Accuracy")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 5-c. CLASSIFICATION: EXAMPLE PREDICTION FOR A NEW MOVIE
# -----------------------------------------------
# new_movie gets created from Regression example prediction above.

# 1st Movie
dt_class_new_pred1 = dt_classifier.predict(new_movie_1)[0]
nb_class_new_pred1 = nb_classifier.predict(new_movie_1)[0]

# 2nd Movie
dt_class_new_pred2 = dt_classifier.predict(new_movie_2)[0]
nb_class_new_pred2 = nb_classifier.predict(new_movie_2)[0]

print("\n==============================")
print("Hit or Flop Prediction")
print("==============================")
print("1st Movie")
if dt_class_new_pred1 == 1:
    print("Decision Tree prediction: Hit")
else:
    print("Decision Tree prediction: Flop")
if nb_class_new_pred1 == 1:
    print("Naive Bayes prediction: Hit")
else:
    print("Naive Bayes prediction: Flop")

print("\n2st Movie")
if dt_class_new_pred2 == 1:
    print("Decision Tree prediction: Hit")
else:
    print("Decision Tree prediction: Flop")
if nb_class_new_pred2 == 1:
    print("Naive Bayes prediction: Hit")
else:
    print("Naive Bayes prediction: Flop")
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
