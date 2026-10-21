"""
Test suite for InsurePredict AI
Validates Flask routes, input validation, feature engineering,
model inference, schema integrity, and model artifact availability.
"""

import json
import os
import sys
import pytest

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app, MODEL_PATH, prepare_features, validate_input_data


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_homepage_loads(client):
    """Verify that the home page loads successfully with 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"InsurePredict" in response.data
    assert b"Calculate My Insurance" in response.data


def test_health_endpoint(client):
    """Verify that the /health endpoint responds with JSON and valid health status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data
    assert data["status"] in ["healthy", "degraded"]


def test_feature_engineering_math():
    """Verify exact formula for interaction and polynomial terms."""
    cleaned_data = {
        "age": 30,
        "bmi": 25.5,
        "children": 2,
        "smoker": 1,
        "sex": 0,
        "region": "northeast",
    }
    df, row = prepare_features(cleaned_data)

    # Check exact values
    assert row["age_bmi"] == 30 * 25.5  # 765.0
    assert row["smoker_bmi"] == 1 * 25.5  # 25.5
    assert row["age_squared"] == 30 ** 2  # 900
    assert row["bmi_squared"] == 25.5 ** 2  # 650.25

    # Check non-smoker interaction term is zero
    cleaned_non_smoker = cleaned_data.copy()
    cleaned_non_smoker["smoker"] = 0
    _, row_ns = prepare_features(cleaned_non_smoker)
    assert row_ns["smoker_bmi"] == 0.0


def test_validation_rejects_negative_and_out_of_range_values():
    """Verify input validation rejects invalid ages, BMIs, and out-of-range counts."""
    # Negative age
    valid, errors = validate_input_data({"age": -5, "bmi": 22, "smoker": "no", "sex": "male"})
    assert not valid
    assert "age" in errors

    # Out of range age (> 120)
    valid, errors = validate_input_data({"age": 145, "bmi": 22, "smoker": "no", "sex": "male"})
    assert not valid
    assert "age" in errors

    # Out of range BMI (< 10 or > 75)
    valid, errors = validate_input_data({"age": 30, "bmi": 5.0, "smoker": "no", "sex": "male"})
    assert not valid
    assert "bmi" in errors

    # Invalid smoker option
    valid, errors = validate_input_data({"age": 30, "bmi": 24.0, "smoker": "sometimes", "sex": "male"})
    assert not valid
    assert "smoker" in errors

    # Invalid sex option
    valid, errors = validate_input_data({"age": 30, "bmi": 24.0, "smoker": "no", "sex": "unknown"})
    assert not valid
    assert "sex" in errors


def test_predict_validation_error_response(client):
    """Verify that POST /predict returns 422 with structured field errors on invalid inputs."""
    payload = {
        "age": "invalid_age",
        "bmi": -10,
        "children": 1,
        "smoker": "no",
        "sex": "male",
        "region": "northeast",
    }
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    # If model is missing, 503 is returned first; if model is loaded, 422 is returned
    assert response.status_code in [422, 503]
    data = response.get_json()
    assert data["success"] is False


def test_predict_behavior_with_or_without_model(client):
    """
    Test behavior under model availability:
    - If model artifact is missing: verifies 503 with helpful setup instructions.
    - If model artifact is present: verifies 200 with accurate numeric prediction.
    """
    valid_payload = {
        "age": 35,
        "sex": "male",
        "bmi": 26.5,
        "children": 1,
        "smoker": "no",
        "region": "northwest",
    }

    response = client.post(
        "/predict",
        data=json.dumps(valid_payload),
        content_type="application/json",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )

    data = response.get_json()

    if not os.path.exists(MODEL_PATH):
        # Explicit test requirement: clearly report model artifact is required
        assert response.status_code == 503
        assert data["success"] is False
        assert "Model artifact not loaded" in data["error"]
        print("\n[NOTE] Model artifact is missing as expected. Awaiting export from notebook.")
    else:
        # Model is present
        assert response.status_code == 200
        assert data["success"] is True
        assert isinstance(data["prediction"], (int, float))
        assert data["prediction"] > 0
        assert "prediction_formatted" in data
        assert "engineered_features" in data
        assert data["engineered_features"]["age_bmi"] == round(35 * 26.5, 2)
        assert data["engineered_features"]["smoker_bmi"] == 0.0  # non-smoker


def test_smoker_risk_premium_multiplier(client):
    """Verify that a smoker has significantly higher predicted insurance cost than a non-smoker."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model not available yet.")

    base_profile = {
        "age": 40,
        "sex": "female",
        "bmi": 32.0,  # High BMI
        "children": 1,
        "region": "southeast",
    }

    # Non-smoker
    resp_ns = client.post(
        "/predict",
        data=json.dumps({**base_profile, "smoker": "no"}),
        content_type="application/json",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    pred_ns = resp_ns.get_json()["prediction"]

    # Smoker
    resp_s = client.post(
        "/predict",
        data=json.dumps({**base_profile, "smoker": "yes"}),
        content_type="application/json",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    pred_s = resp_s.get_json()["prediction"]

    # Smoker should be much higher due to smoker_bmi interaction term and base risk
    assert pred_s > pred_ns
    assert pred_s >= pred_ns * 1.5


def test_model_info_endpoint(client):
    """Verify /api/model-info returns metadata and hyperparameters."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.get_json()
    assert "model_loaded" in data
    assert "metadata" in data
    metadata = data["metadata"]
    assert metadata.get("algorithm") == "XGBoost Regressor (Optimized)"
    assert metadata["hyperparameters"]["n_estimators"] == 300
    assert metadata["hyperparameters"]["learning_rate"] == 0.05
    assert metadata["hyperparameters"]["max_depth"] == 4
    assert metadata["hyperparameters"]["random_state"] == 42


def test_traditional_html_form_submission(client):
    """Verify that traditional form POST (without AJAX) renders result.html."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model not available yet.")

    form_data = {
        "age": "28",
        "sex": "female",
        "bmi": "22.4",
        "children": "0",
        "smoker": "no",
        "region": "northeast",
    }
    response = client.post("/predict", data=form_data)
    assert response.status_code == 200
    assert b"Estimated Annual Medical Charges" in response.data
    assert b"Submitted Profile" in response.data

