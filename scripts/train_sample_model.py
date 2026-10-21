"""
Training script matching the user's exact notebook training pipeline.
Can be executed if the user provides 'insurance.csv' or wants to train
the XGBoost model locally with the exact hyperparameters:
- n_estimators = 300
- learning_rate = 0.05
- max_depth = 4
- random_state = 42
- Feature engineering:
    age_bmi = age * bmi
    smoker_bmi = smoker * bmi
    age_squared = age ** 2
    bmi_squared = bmi ** 2
"""

import argparse
import json
import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_or_fetch_dataset(csv_path=None):
    if csv_path and os.path.exists(csv_path):
        print(f"Loading dataset from: {csv_path}")
        return pd.read_csv(csv_path)

    # Check common default locations
    default_locations = [
        os.path.join(PROJECT_ROOT, "insurance.csv"),
        os.path.join(PROJECT_ROOT, "data", "insurance.csv"),
    ]
    for loc in default_locations:
        if os.path.exists(loc):
            print(f"Loading dataset from: {loc}")
            return pd.read_csv(loc)

    print("Attempting to fetch benchmark medical insurance dataset...")
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    local_target = os.path.join(data_dir, "insurance.csv")
    urllib.request.urlretrieve(url, local_target)
    print(f"Downloaded benchmark dataset to {local_target}")
    return pd.read_csv(local_target)


def train_and_export(csv_path=None, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(output_dir, exist_ok=True)
    df = load_or_fetch_dataset(csv_path)

    print(f"Dataset Shape: {df.shape}")
    print(f"Dataset Columns: {list(df.columns)}")

    # Target
    target_col = "charges" if "charges" in df.columns else df.columns[-1]
    Y = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # Standard Preprocessing matching standard practice:
    # smoker: yes -> 1, no -> 0
    # sex: female -> 0, male -> 1
    # region: one-hot or categorical encoding
    X = X_raw.copy()
    if "smoker" in X.columns and not pd.api.types.is_numeric_dtype(X["smoker"]):
        X["smoker"] = X["smoker"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)
    if "sex" in X.columns and not pd.api.types.is_numeric_dtype(X["sex"]):
        X["sex"] = X["sex"].astype(str).str.strip().str.lower().map({"male": 1, "female": 0}).fillna(0).astype(int)

    # One-hot encode region if present and categorical/string
    if "region" in X.columns and not pd.api.types.is_numeric_dtype(X["region"]):
        X = pd.get_dummies(X, columns=["region"], drop_first=False, dtype=int)

    # --- 2. PREPARE & ENGINEER FEATURES ---
    # Exact code from the user's notebook specification:
    X_engineered = X.copy()
    X_engineered["age_bmi"] = X_engineered["age"] * X_engineered["bmi"]
    X_engineered["smoker_bmi"] = X_engineered["smoker"] * X_engineered["bmi"]
    X_engineered["age_squared"] = X_engineered["age"] ** 2
    X_engineered["bmi_squared"] = X_engineered["bmi"] ** 2

    # --- 3. SPLIT THE DATASET ---
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_engineered, Y, test_size=0.2, random_state=42
    )

    # --- 4. TRAIN MODELS ---
    print("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, Y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(Y_test, rf_pred)
    print(f"Random Forest R2 Score: {rf_r2:.4f}")

    print("Training Optimized XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    xgb_model.fit(X_train, Y_train)
    Y_pred = xgb_model.predict(X_test)

    # --- 5. EVALUATE XGBOOST PERFORMANCE ---
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)

    print("\n--- XGBoost Performance Metrics ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # --- 6. EXPORT ARTIFACTS ---
    model_path = os.path.join(output_dir, "xgb_model.pkl")
    metadata_path = os.path.join(output_dir, "model_metadata.json")

    joblib.dump(xgb_model, model_path)
    print(f"\n[OK] Saved XGBoost model to {model_path}")

    metadata = {
        "model_filename": "xgb_model.pkl",
        "algorithm": "XGBoost Regressor (Optimized)",
        "hyperparameters": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "random_state": 42,
        },
        "metrics": {
            "r2_score": round(float(r2), 4),
            "mae": round(float(mae), 2),
            "mse": round(float(mse), 2),
            "rmse": round(float(rmse), 2),
            "rf_r2_score": round(float(rf_r2), 4),
        },
        "feature_order": list(X_train.columns),
        "engineered_features": [
            "age_bmi",
            "smoker_bmi",
            "age_squared",
            "bmi_squared",
        ],
        "base_features": [
            "age",
            "sex",
            "bmi",
            "children",
            "smoker",
            "region",
        ],
        "feature_encodings": {
            "sex": {"type": "binary_numeric", "mapping": {"female": 0, "male": 1}},
            "smoker": {"type": "binary_numeric", "mapping": {"no": 0, "yes": 1}},
            "region": {
                "type": "one_hot",
                "categories": ["northeast", "northwest", "southeast", "southwest"],
                "columns": [col for col in X_train.columns if col.startswith("region_")],
            },
        },
        "currency": "USD",
        "currency_symbol": "$",
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved model metadata to {metadata_path}")
    return xgb_model, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export XGBoost insurance model.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to insurance.csv")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    train_and_export(csv_path=args.data_path, output_dir=args.output_dir)
