"""
InsurePredict AI - Backend Application
Flask REST API and Web Interface for AI-powered Medical Insurance Cost Prediction.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("insurepredict")

app = Flask(__name__)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
JSON_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.json")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Global model state
ML_MODEL = None
MODEL_METADATA = {}


def load_model_and_metadata():
    """Loads the trained model with multi-tier fallback for 100% cross-platform portability."""
    global ML_MODEL, MODEL_METADATA

    # 1. Load Metadata if available
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                MODEL_METADATA = json.load(f)
            logger.info("Loaded model metadata successfully.")
        except Exception as e:
            logger.error("Failed to parse model_metadata.json: %s", str(e))
            MODEL_METADATA = {}
    else:
        logger.warning("model_metadata.json not found at %s", METADATA_PATH)
        MODEL_METADATA = {}

    # 2. Resilient Model Loading across platforms
    loaded = False

    # Tier 1: Load XGBoost via JSON format (native, version-independent, cross-OS)
    if os.path.exists(JSON_MODEL_PATH):
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor()
            model.load_model(JSON_MODEL_PATH)
            ML_MODEL = model
            MODEL_METADATA["model_status"] = "loaded"
            MODEL_METADATA["active_algorithm"] = "XGBoost Regressor (JSON)"
            logger.info("Successfully loaded XGBoost model from %s", JSON_MODEL_PATH)
            loaded = True
        except Exception as e:
            logger.warning("Tier 1 JSON load skipped: %s", str(e))

    # Tier 2: Load XGBoost via Pickle/Joblib
    if not loaded and os.path.exists(MODEL_PATH):
        try:
            ML_MODEL = joblib.load(MODEL_PATH)
            MODEL_METADATA["model_status"] = "loaded"
            MODEL_METADATA["active_algorithm"] = "XGBoost Regressor (Pickle)"
            logger.info("Successfully loaded XGBoost model from %s", MODEL_PATH)
            loaded = True
        except Exception as e:
            logger.warning("Tier 2 Pickle load skipped: %s", str(e))

    # Tier 3: Pure Scikit-Learn Random Forest fallback (requires no C/OpenMP dylibs)
    if not loaded and os.path.exists(RF_MODEL_PATH):
        try:
            ML_MODEL = joblib.load(RF_MODEL_PATH)
            MODEL_METADATA["model_status"] = "loaded"
            MODEL_METADATA["active_algorithm"] = "Random Forest Regressor (Fallback)"
            logger.info("Successfully loaded fallback Random Forest model from %s", RF_MODEL_PATH)
            loaded = True
        except Exception as e:
            logger.warning("Tier 3 Fallback load skipped: %s", str(e))

    if not loaded:
        logger.warning("No model artifact could be loaded. Application starting in awaiting mode.")
        ML_MODEL = None
        MODEL_METADATA["model_status"] = "awaiting_artifact"


# Attempt to load model on module import
load_model_and_metadata()


def validate_input_data(data):
    """
    Validates form or JSON inputs against expected medical constraints.
    Returns: (is_valid, cleaned_data_or_error_dict)
    """
    errors = {}
    cleaned = {}

    # 1. Age Validation
    raw_age = data.get("age")
    if raw_age is None or str(raw_age).strip() == "":
        errors["age"] = "Age is required."
    else:
        try:
            age = float(raw_age)
            if not age.is_integer() or age < 1 or age > 120:
                errors["age"] = "Age must be a whole number between 1 and 120."
            else:
                cleaned["age"] = int(age)
        except (ValueError, TypeError):
            errors["age"] = "Age must be a valid number."

    # 2. BMI Validation
    raw_bmi = data.get("bmi")
    if raw_bmi is None or str(raw_bmi).strip() == "":
        errors["bmi"] = "BMI is required."
    else:
        try:
            bmi = float(raw_bmi)
            if bmi < 10.0 or bmi > 75.0:
                errors["bmi"] = "BMI must be between 10.0 and 75.0."
            else:
                cleaned["bmi"] = round(bmi, 2)
        except (ValueError, TypeError):
            errors["bmi"] = "BMI must be a valid decimal number."

    # 3. Children Validation
    raw_children = data.get("children", 0)
    if raw_children is None or str(raw_children).strip() == "":
        cleaned["children"] = 0
    else:
        try:
            children = float(raw_children)
            if not children.is_integer() or children < 0 or children > 20:
                errors["children"] = "Number of dependents must be an integer between 0 and 20."
            else:
                cleaned["children"] = int(children)
        except (ValueError, TypeError):
            errors["children"] = "Dependents must be a valid integer."

    # 4. Smoker Validation
    raw_smoker = str(data.get("smoker", "")).strip().lower()
    if raw_smoker in ["1", "true", "yes", "y"]:
        cleaned["smoker"] = 1
        cleaned["smoker_display"] = "Yes"
    elif raw_smoker in ["0", "false", "no", "n"]:
        cleaned["smoker"] = 0
        cleaned["smoker_display"] = "No"
    else:
        errors["smoker"] = "Smoking status is required (Yes or No)."

    # 5. Sex Validation
    raw_sex = str(data.get("sex", "")).strip().lower()
    if raw_sex in ["1", "male", "m"]:
        cleaned["sex"] = 1
        cleaned["sex_display"] = "Male"
    elif raw_sex in ["0", "female", "f"]:
        cleaned["sex"] = 0
        cleaned["sex_display"] = "Female"
    else:
        errors["sex"] = "Sex is required (Male or Female)."

    # 6. Region Validation
    valid_regions = ["northeast", "northwest", "southeast", "southwest"]
    raw_region = str(data.get("region", "")).strip().lower()
    if raw_region in valid_regions:
        cleaned["region"] = raw_region
        cleaned["region_display"] = raw_region.capitalize()
    elif not raw_region:
        # Default or required based on base features
        cleaned["region"] = "northeast"
        cleaned["region_display"] = "Northeast"
    else:
        errors["region"] = f"Invalid region. Must be one of: {', '.join(valid_regions)}."

    if errors:
        return False, errors
    return True, cleaned


def prepare_features(cleaned):
    """
    Transforms user inputs using exact training feature pipeline and engineering:
    - Interaction features: age_bmi, smoker_bmi
    - Polynomial features: age_squared, bmi_squared
    - Region one-hot or categorical encoding
    - Strict column order matching the trained XGBoost model
    """
    age = cleaned["age"]
    bmi = cleaned["bmi"]
    children = cleaned["children"]
    smoker = cleaned["smoker"]
    sex = cleaned["sex"]
    region = cleaned["region"]

    # Calculate exact interaction & polynomial features as specified in notebook
    age_bmi = age * bmi
    smoker_bmi = smoker * bmi
    age_squared = age ** 2
    bmi_squared = bmi ** 2

    # Expected feature order
    feature_order = MODEL_METADATA.get("feature_order")
    if not feature_order and ML_MODEL is not None and hasattr(ML_MODEL, "feature_names_in_"):
        feature_order = list(ML_MODEL.feature_names_in_)

    if not feature_order:
        # Default fallback feature ordering
        feature_order = [
            "age",
            "sex",
            "bmi",
            "children",
            "smoker",
            "region_northeast",
            "region_northwest",
            "region_southeast",
            "region_southwest",
            "age_bmi",
            "smoker_bmi",
            "age_squared",
            "bmi_squared",
        ]

    # Build row dictionary
    row = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "age_bmi": age_bmi,
        "smoker_bmi": smoker_bmi,
        "age_squared": age_squared,
        "bmi_squared": bmi_squared,
    }

    # Region one-hot mapping
    regions = ["northeast", "northwest", "southeast", "southwest"]
    for r in regions:
        col_name = f"region_{r}"
        row[col_name] = 1 if region == r else 0

    # If region is encoded as an integer in the model
    if "region" in feature_order and "region_northeast" not in feature_order:
        region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
        row["region"] = region_map.get(region, 0)

    # Reorder according to model schema
    ordered_values = {}
    for col in feature_order:
        if col in row:
            ordered_values[col] = [row[col]]
        else:
            logger.warning("Feature %s not in generated row, filling with 0", col)
            ordered_values[col] = [0]

    feature_df = pd.DataFrame(ordered_values)
    return feature_df, row


@app.route("/", methods=["GET"])
def index():
    """Renders the main InsurePredict AI landing page."""
    currency_symbol = MODEL_METADATA.get("currency_symbol", "$")
    model_loaded = ML_MODEL is not None
    metrics = MODEL_METADATA.get("metrics", {})
    return render_template(
        "index.html",
        model_loaded=model_loaded,
        currency_symbol=currency_symbol,
        metrics=metrics,
        metadata=MODEL_METADATA,
    )


@app.route("/health", methods=["GET"])
def health():
    """Returns application health status and model availability."""
    model_loaded = ML_MODEL is not None
    status = "healthy" if model_loaded else "degraded"
    return (
        jsonify(
            {
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_loaded": model_loaded,
                "model_algorithm": MODEL_METADATA.get("algorithm", "XGBoost Regressor"),
                "features_expected": len(MODEL_METADATA.get("feature_order", [])),
                "metrics_available": bool(MODEL_METADATA.get("metrics", {}).get("r2_score")),
            }
        ),
        200 if model_loaded else 200,
    )


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Returns detailed model metadata, hyperparameters, and evaluation metrics."""
    return jsonify(
        {
            "model_loaded": ML_MODEL is not None,
            "metadata": MODEL_METADATA,
            "model_path": MODEL_PATH,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Validates form data or JSON, generates inference via XGBoost model,
    and returns estimated medical insurance cost.
    """
    # 1. Check if model is loaded
    if ML_MODEL is None:
        load_model_and_metadata()  # Re-attempt hot reload if user just copied the model
        if ML_MODEL is None:
            err_msg = (
                "Model artifact not loaded. Please ensure 'models/xgb_model.pkl' exists. "
                "You can export it from your notebook using 'scripts/export_from_notebook.py' "
                "or train one with 'python scripts/train_sample_model.py'."
            )
            logger.error("Prediction attempted without model artifact.")
            if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"success": False, "error": err_msg}), 503
            return (
                render_template(
                    "result.html",
                    success=False,
                    error=err_msg,
                    currency_symbol=MODEL_METADATA.get("currency_symbol", "$"),
                ),
                503,
            )

    # 2. Extract payload
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()

    if not data:
        return (
            jsonify({"success": False, "error": "No input data provided in request."}),
            400,
        )

    # 3. Validate input
    is_valid, validation_result = validate_input_data(data)
    if not is_valid:
        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"success": False, "validation_errors": validation_result}), 422
        return (
            render_template(
                "index.html",
                validation_errors=validation_result,
                form_data=data,
                model_loaded=True,
                currency_symbol=MODEL_METADATA.get("currency_symbol", "$"),
            ),
            422,
        )

    cleaned = validation_result

    # 4. Feature engineering and order mapping
    try:
        features_df, full_features = prepare_features(cleaned)
    except Exception as e:
        logger.exception("Error during feature preparation: %s", str(e))
        return (
            jsonify({"success": False, "error": f"Feature engineering failed: {str(e)}"}),
            500,
        )

    # 5. Model Inference
    try:
        prediction_val = float(ML_MODEL.predict(features_df)[0])
        # Insurance charges cannot be negative; guard against minor tree boundary artifacts
        prediction_val = max(0.0, prediction_val)
        prediction_formatted = f"{prediction_val:,.2f}"
    except Exception as e:
        logger.exception("Inference failed: %s", str(e))
        return (
            jsonify({"success": False, "error": f"Prediction computation failed: {str(e)}"}),
            500,
        )

    currency_symbol = MODEL_METADATA.get("currency_symbol", "$")
    currency = MODEL_METADATA.get("currency", "USD")

    # BMI Category Calculation
    bmi = cleaned["bmi"]
    if bmi < 18.5:
        bmi_category = "Underweight"
        bmi_risk = "Low-to-moderate risk"
    elif bmi < 25.0:
        bmi_category = "Normal Weight"
        bmi_risk = "Standard baseline risk"
    elif bmi < 30.0:
        bmi_category = "Overweight"
        bmi_risk = "Elevated risk"
    else:
        bmi_category = "Obese"
        bmi_risk = "High risk factor"

    # Risk factor impact assessment
    smoker_impact = (
        "High impact - smoking significantly multiplies premium cost, especially with elevated BMI."
        if cleaned["smoker"] == 1
        else "Low impact - non-smoker discount applied."
    )

    response_payload = {
        "success": True,
        "prediction": round(prediction_val, 2),
        "prediction_formatted": f"{currency_symbol}{prediction_formatted}",
        "currency": currency,
        "currency_symbol": currency_symbol,
        "user_inputs": {
            "age": cleaned["age"],
            "sex": cleaned["sex_display"],
            "bmi": cleaned["bmi"],
            "bmi_category": bmi_category,
            "bmi_risk": bmi_risk,
            "children": cleaned["children"],
            "smoker": cleaned["smoker_display"],
            "region": cleaned["region_display"],
        },
        "engineered_features": {
            "age_bmi": round(float(full_features["age_bmi"]), 2),
            "smoker_bmi": round(float(full_features["smoker_bmi"]), 2),
            "age_squared": round(float(full_features["age_squared"]), 2),
            "bmi_squared": round(float(full_features["bmi_squared"]), 2),
        },
        "insights": {
            "smoker_impact": smoker_impact,
            "bmi_impact": f"{bmi_category} ({bmi}) - {bmi_risk}",
            "model_used": MODEL_METADATA.get("algorithm", "XGBoost Regressor"),
        },
    }

    # If submitted via traditional form POST without AJAX
    if not request.is_json and request.headers.get("X-Requested-With") != "XMLHttpRequest":
        return render_template("result.html", result=response_payload)

    return jsonify(response_payload), 200


def find_available_port(start_port=5001, max_attempts=50):
    """Finds an available TCP port to prevent Address already in use errors."""
    import socket
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return start_port


if __name__ == "__main__":
    import threading
    import webbrowser

    preferred_port = int(os.environ.get("PORT", 5001))
    port = find_available_port(preferred_port)
    host = os.environ.get("HOST", "0.0.0.0")

    url = f"http://localhost:{port}"
    logger.info("Starting InsurePredict AI server on %s (and http://127.0.0.1:%d)", url, port)

    # Automatically launch default browser for zero-friction experience
    if os.environ.get("NO_BROWSER", "").lower() not in ["1", "true"]:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port, debug=False)
