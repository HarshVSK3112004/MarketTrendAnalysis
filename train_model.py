# ==========================================
# Market Trend Analysis
# Train Machine Learning Model
# Part 1
# ==========================================

import os
import joblib
import warnings

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ==========================================
# Load Dataset
# ==========================================

DATASET_PATH = "dataset/market_data.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        "dataset/market_data.csv not found.\nRun download_data.py first."
    )

data = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

print("Rows :", len(data))
print("Columns :", len(data.columns))

print(data.head())

# ==========================================
# Convert Numeric Columns
# ==========================================

numeric_cols = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for col in numeric_cols:

    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )

# ==========================================
# Sort Data
# ==========================================

data = data.sort_values(
    by=[
        "Stock",
        "Date"
    ]
)

data.reset_index(
    drop=True,
    inplace=True
)

# ==========================================
# Technical Indicators
# ==========================================

print("\nCalculating Indicators...")

# SMA

data["SMA20"] = (
    data
    .groupby("Stock")["Close"]
    .transform(
        lambda x: x.rolling(20).mean()
    )
)

data["SMA50"] = (
    data
    .groupby("Stock")["Close"]
    .transform(
        lambda x: x.rolling(50).mean()
    )
)

# EMA

data["EMA20"] = (
    data
    .groupby("Stock")["Close"]
    .transform(
        lambda x: x.ewm(span=20).mean()
    )
)

data["EMA50"] = (
    data
    .groupby("Stock")["Close"]
    .transform(
        lambda x: x.ewm(span=50).mean()
    )
)

# ==========================================
# Price Features
# ==========================================

data["Price_Change"] = (
    data["Close"] -
    data["Open"]
)

data["Daily_Return"] = (
    data
    .groupby("Stock")["Close"]
    .pct_change()
)

data["Return_5D"] = (
    data
    .groupby("Stock")["Close"]
    .pct_change(5)
)

data["Return_10D"] = (
    data
    .groupby("Stock")["Close"]
    .pct_change(10)
)

# ==========================================
# Volume Features
# ==========================================

data["Volume_Change"] = (
    data
    .groupby("Stock")["Volume"]
    .pct_change()
)

data["Prev_Volume"] = (
    data
    .groupby("Stock")["Volume"]
    .shift(1)
)

# ==========================================
# Previous Day Values
# ==========================================

data["Prev_Open"] = (
    data
    .groupby("Stock")["Open"]
    .shift(1)
)

data["Prev_High"] = (
    data
    .groupby("Stock")["High"]
    .shift(1)
)

data["Prev_Low"] = (
    data
    .groupby("Stock")["Low"]
    .shift(1)
)

data["Prev_Close"] = (
    data
    .groupby("Stock")["Close"]
    .shift(1)
)

# ==========================================
# Volatility
# ==========================================

data["Volatility"] = (
    data["High"] -
    data["Low"]
)

# ==========================================
# RSI
# ==========================================

delta = (
    data
    .groupby("Stock")["Close"]
    .diff()
)

gain = delta.where(
    delta > 0,
    0
)

loss = -delta.where(
    delta < 0,
    0
)

avg_gain = (
    gain
    .groupby(data["Stock"])
    .transform(
        lambda x: x.rolling(14).mean()
    )
)

avg_loss = (
    loss
    .groupby(data["Stock"])
    .transform(
        lambda x: x.rolling(14).mean()
    )
)

rs = avg_gain / avg_loss

data["RSI"] = (
    100 -
    (100 / (1 + rs))
)

# ==========================================
# MACD
# ==========================================

ema12 = (
    data
    .groupby("Stock")["Close"]
    .transform(
        lambda x: x.ewm(span=12).mean()
    )
)

ema26 = (
    data
    .groupby("Stock")["Close"]
    .transform(
        lambda x: x.ewm(span=26).mean()
    )
)

data["MACD"] = ema12 - ema26

data["MACD_Signal"] = (
    data
    .groupby("Stock")["MACD"]
    .transform(
        lambda x: x.ewm(span=9).mean()
    )
)

# ==========================================
# Bollinger Bands
# ==========================================

rolling_mean = (
    data
    .groupby("Stock")["Close"]
    .transform(
        lambda x: x.rolling(20).mean()
    )
)

rolling_std = (
    data
    .groupby("Stock")["Close"]
    .transform(
        lambda x: x.rolling(20).std()
    )
)

data["BB_Upper"] = (
    rolling_mean +
    2 * rolling_std
)

data["BB_Lower"] = (
    rolling_mean -
    2 * rolling_std
)

# ==========================================
# ATR
# ==========================================

high_low = (
    data["High"] -
    data["Low"]
)

high_close = (
    data["High"] -
    data["Prev_Close"]
).abs()

low_close = (
    data["Low"] -
    data["Prev_Close"]
).abs()

true_range = pd.concat(
    [
        high_low,
        high_close,
        low_close
    ],
    axis=1
).max(axis=1)

data["ATR"] = (
    true_range
    .groupby(data["Stock"])
    .transform(
        lambda x: x.rolling(14).mean()
    )
)

# ==========================================
# Target Variable
# ==========================================

data["Trend"] = (
    data
    .groupby("Stock")["Close"]
    .shift(-1)
    >
    data["Close"]
).astype(int)

# ==========================================
# Remove Missing Values
# ==========================================

data.dropna(inplace=True)

data.reset_index(
    drop=True,
    inplace=True
)

print("\nRows after preprocessing :", len(data))

print("\nTrend Distribution")

print(data["Trend"].value_counts())
# ==========================================
# Feature Selection
# ==========================================

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA20",
    "SMA50",
    "EMA20",
    "EMA50",
    "RSI",
    "MACD",
    "MACD_Signal",
    "BB_Upper",
    "BB_Lower",
    "ATR",
    "Price_Change",
    "Volatility",
    "Prev_Open",
    "Prev_High",
    "Prev_Low",
    "Prev_Close",
    "Prev_Volume",
    "Daily_Return",
    "Return_5D",
    "Return_10D",
    "Volume_Change"
]

X = data[FEATURES]

y = data["Trend"]

print("\nFeature Count :", len(FEATURES))
print("Training Samples :", len(X))

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining :", len(X_train))
print("Testing  :", len(X_test))

# ==========================================
# Random Forest Model
# ==========================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=18,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Training Completed")

# ==========================================
# Prediction
# ==========================================

pred = model.predict(X_test)

prob = model.predict_proba(X_test)

# ==========================================
# Evaluation Metrics
# ==========================================

accuracy = accuracy_score(
    y_test,
    pred
)

precision = precision_score(
    y_test,
    pred
)

recall = recall_score(
    y_test,
    pred
)

f1 = f1_score(
    y_test,
    pred
)

print("\n")
print("="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        pred
    )
)

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    pred
)

print("\nConfusion Matrix")

print(cm)

# ==========================================
# Feature Importance
# ==========================================

importance = pd.DataFrame({

    "Feature": FEATURES,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n")
print("="*60)
print("FEATURE IMPORTANCE")
print("="*60)

print(importance)

# ==========================================
# Top 10 Features
# ==========================================

print("\nTop 10 Important Features\n")

print(
    importance.head(10)
)

# ==========================================
# Save Model
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/trend_model.pkl"
)

print("\nModel Saved Successfully")

print("Location : models/trend_model.pkl")
# ==========================================
# PART 3
# Save Reports & Visualizations
# ==========================================

import matplotlib.pyplot as plt

print("\nGenerating Reports...")

# ==========================================
# Create reports folder
# ==========================================

os.makedirs(
    "reports",
    exist_ok=True
)

# ==========================================
# Save Feature Importance CSV
# ==========================================

importance.to_csv(
    "reports/feature_importance.csv",
    index=False
)

print("Feature Importance CSV Saved")

# ==========================================
# Save Feature Importance Plot
# ==========================================

plt.figure(figsize=(12,8))

top_features = importance.head(15)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 15 Important Features")

plt.xlabel("Importance")

plt.tight_layout()

plt.savefig(
    "reports/feature_importance.png"
)

plt.close()

print("Feature Importance Graph Saved")

# ==========================================
# Save Metrics
# ==========================================

with open(
    "reports/model_report.txt",
    "w"
) as f:

    f.write("="*50 + "\n")
    f.write("MARKET TREND ANALYSIS MODEL REPORT\n")
    f.write("="*50 + "\n\n")

    f.write(f"Rows Used : {len(data)}\n")
    f.write(f"Features  : {len(FEATURES)}\n\n")

    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1 Score  : {f1:.4f}\n\n")

    f.write("Confusion Matrix\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write("Classification Report\n")
    f.write(
        classification_report(
            y_test,
            pred
        )
    )

print("Model Report Saved")

# ==========================================
# Save Feature Names
# ==========================================

joblib.dump(
    FEATURES,
    "models/feature_names.pkl"
)

print("Feature Names Saved")

# ==========================================
# Cross Validation
# ==========================================

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("\nCross Validation Scores")

print(scores)

print(
    "Average CV Accuracy:",
    round(scores.mean(),4)
)

# ==========================================
# Training Summary
# ==========================================

print("\n")
print("="*70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("="*70)

print(f"Dataset Rows          : {len(data)}")
print(f"Training Samples      : {len(X_train)}")
print(f"Testing Samples       : {len(X_test)}")
print(f"Features Used         : {len(FEATURES)}")

print()

print(f"Accuracy              : {accuracy:.4f}")
print(f"Precision             : {precision:.4f}")
print(f"Recall                : {recall:.4f}")
print(f"F1 Score              : {f1:.4f}")

print()

print(f"Cross Validation Mean : {scores.mean():.4f}")

print()

print("Saved Files")

print("✓ models/trend_model.pkl")
print("✓ models/feature_names.pkl")
print("✓ reports/model_report.txt")
print("✓ reports/feature_importance.csv")
print("✓ reports/feature_importance.png")

print("="*70)