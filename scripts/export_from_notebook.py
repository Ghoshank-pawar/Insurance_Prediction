"""
Export snippet to be copied into the final cell of your Jupyter Notebook.
After training your XGBoost model and evaluating its metrics, run this code.
It will safely serialize your trained XGBoost model and save its exact feature schema,
order, categorical encodings, and performance metrics into 'models/'.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd


def export_notebook_model(
    xgb_estimator=None,
    X_train_df=None,
    X_original_df=None,
    metrics_dict=None,
    output_dir="models",
    currency="USD",
    currency_symbol="$",
):
    """Saves the trained model and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "xgb_model.pkl")
    metadata_path = os.path.join(output_dir, "model_metadata.json")

    # 1. Verify model
    if xgb_estimator is None:
        raise ValueError("Please pass your trained xgb_model instance.")

    # 2. Extract feature names in exact order
    if X_train_df is not None and hasattr(X_train_df, "columns"):
        feature_names = list(X_train_df.columns)
    elif hasattr(xgb_estimator, "feature_names_in_"):
        feature_names = list(xgb_estimator.feature_names_in_)
    else:
        raise ValueError(
            "Could not determine feature names. Please pass X_train (with column names)."
        )

    # 3. Save model
    joblib.dump(xgb_estimator, model_path)
    print(f" [SUCCESS] Saved trained model to: {model_path}")

    # 4. Extract categorical encodings from X if available
    encodings = {}
    if X_original_df is not None and hasattr(X_original_df, "columns"):
        for col in X_original_df.columns:
            if X_original_df[col].dtype == "object" or pd.api.types.is_categorical_dtype(
                X_original_df[col]
            ):
                encodings[col] = {
                    "type": "categorical_string",
                    "unique_values": [str(v) for v in X_original_df[col].unique().tolist()],
                }
            elif set(X_original_df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                encodings[col] = {
                    "type": "binary_numeric",
                    "unique_values": [int(v) for v in X_original_df[col].unique().tolist()],
                }
            else:
                encodings[col] = {
                    "type": "continuous_numeric",
                    "min": float(X_original_df[col].min()),
                    "max": float(X_original_df[col].max()),
                }

    # 5. Extract hyperparameters
    hyperparameters = {}
    for param in ["n_estimators", "learning_rate", "max_depth", "random_state"]:
        if hasattr(xgb_estimator, param):
            hyperparameters[param] = getattr(xgb_estimator, param)
        elif hasattr(xgb_estimator, "get_params"):
            hyperparameters[param] = xgb_estimator.get_params().get(param)

    # 6. Prepare metadata payload
    metadata = {
        "model_filename": "xgb_model.pkl",
        "algorithm": "XGBoost Regressor (Optimized)",
        "hyperparameters": hyperparameters,
        "metrics": metrics_dict or {},
        "feature_order": feature_names,
        "engineered_features": [
            "age_bmi",
            "smoker_bmi",
            "age_squared",
            "bmi_squared",
        ],
        "base_features": list(X_original_df.columns)
        if X_original_df is not None and hasattr(X_original_df, "columns")
        else None,
        "feature_encodings": encodings,
        "currency": currency,
        "currency_symbol": currency_symbol,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f" [SUCCESS] Saved model metadata to: {metadata_path}")
    print("=" * 60)
    print("Your model artifacts are ready for the Flask Web Application!")
    print(f"Features expected ({len(feature_names)}): {feature_names}")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage inside your Jupyter notebook:
    #
    # from scripts.export_from_notebook import export_notebook_model
    # export_notebook_model(
    #     xgb_estimator=xgb_model,
    #     X_train_df=X_train,
    #     X_original_df=X,
    #     metrics_dict={
    #         "r2_score": float(r2),
    #         "mae": float(mae),
    #         "mse": float(mse),
    #         "rmse": float(rmse)
    #     },
    #     output_dir="models"
    # )
    print("Run this function inside your Jupyter Notebook to export your trained model.")
