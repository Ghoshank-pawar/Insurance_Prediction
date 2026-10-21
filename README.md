# InsurePredict AI: AI-Powered Medical Insurance Cost Prediction

InsurePredict AI is a production-grade, responsive full-stack web application that predicts personal medical insurance charges using an optimized **XGBoost Regressor** with domain-specific feature engineering.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Machine Learning Architecture](#machine-learning-architecture)
- [Project Directory Structure](#project-directory-structure)
- [How to Export Your Model from Jupyter Notebook](#how-to-export-your-model-from-jupyter-notebook)
- [Quick Start Guide for macOS](#quick-start-guide-for-macos)
  - [1. Navigate to Project Directory](#1-navigate-to-project-directory)
  - [2. Create & Activate Virtual Environment](#2-create--activate-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Model Artifact Setup](#4-model-artifact-setup)
  - [5. Run Automated Tests](#5-run-automated-tests)
  - [6. Start the Flask Server](#6-start-the-flask-server)
  - [7. Open in Browser & Test Predictions](#7-open-in-browser--test-predictions)
  - [8. Stopping the Server](#8-stopping-the-server)
- [Configuring the Python Interpreter in Antigravity / VS Code](#configuring-the-python-interpreter-in-antigravity--vs-code)
- [API Endpoints](#api-endpoints)
- [Validation & Error Handling](#validation--error-handling)
- [Medical & Financial Disclaimer](#medical--financial-disclaimer)

---

## Project Overview

Underwriting individual medical insurance charges involves balancing complex, non-linear interactions across age, BMI, and lifestyle habits. InsurePredict AI transforms raw user profiles into high-accuracy financial forecasts by computing interaction terms and feeding them through 300 gradient-boosted decision trees.

### Key Highlights
- **Modern Healthcare/Fintech Design:** Navy blue and medical teal palette with responsive layout (mobile, tablet, desktop).
- **Exact Feature Engineering:** Implements `age_bmi`, `smoker_bmi`, `age_squared`, and `bmi_squared`.
- **Dual Form Submission:** Seamless asynchronous JSON AJAX with dynamic Chart.js visualizations, plus fallback traditional HTML form POST.
- **Robust Input Validation:** Strict client and server validation rejecting invalid, negative, or out-of-range inputs.
- **Model Decoupling & Graceful Degradation:** Clean decoupling of model artifacts from application code. If `models/xgb_model.pkl` is awaiting export, the UI provides clear instructions and the `/health` endpoint reports `status: degraded`.

---

## Machine Learning Architecture

The prediction engine utilizes the exact hyperparameter specifications and feature engineering from your training notebook:

### Algorithms
1. **XGBoost Regressor (Optimized)** — Primary inference model
2. **Random Forest Regressor** — Benchmark comparison model

### XGBoost Hyperparameters
```python
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
```

### Feature Engineering Pipeline
The backend generates these four interaction and polynomial features before passing data to the model:
- `age_bmi` = $\text{age} \times \text{bmi}$ (Captures accelerating risk of elevated BMI with advancing age)
- `smoker_bmi` = $\text{smoker} \times \text{bmi}$ (Multiplier capturing the severe combined risk of smoking and obesity)
- `age_squared` = $\text{age}^2$ (Captures non-linear aging acceleration)
- `bmi_squared` = $\text{bmi}^2$ (Captures quadratic non-linear obesity risk)

### Evaluated Performance Metrics
- **R² Score:** `0.8777` (~87.8% of variance explained)
- **Mean Absolute Error (MAE):** `$2,409.17`
- **Root Mean Squared Error (RMSE):** `$4,356.82`
- **Random Forest Baseline R²:** `0.8674`

---

## Project Directory Structure

```text
insurance-prediction-app/
├── app.py                      # Flask REST API and web application server
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation & setup instructions
├── data/
│   └── insurance.csv           # Benchmark dataset
├── models/
│   ├── xgb_model.pkl           # Serialized trained XGBoost model artifact
│   └── model_metadata.json     # Feature order, encodings, and performance metrics
├── templates/
│   ├── index.html              # Main responsive landing & interactive calculator page
│   └── result.html             # Standalone result page for traditional form submissions
├── static/
│   ├── css/
│   │   └── style.css           # Custom healthcare & fintech responsive design
│   ├── js/
│   │   └── script.js           # Validation, AJAX fetch, Chart.js benchmarks & BMI helper
│   └── images/
├── scripts/
│   ├── export_from_notebook.py # Helper snippet to paste into your Jupyter Notebook
│   └── train_sample_model.py   # Standalone training script using your exact parameters
└── tests/
    └── test_prediction.py      # Comprehensive pytest test suite (9 test cases)
```

---

## How to Export Your Model from Jupyter Notebook

If you have already trained your model inside your Jupyter Notebook, copy and run the following code in a new cell at the bottom of your notebook:

```python
import os
import json
import joblib

# Target folder in the web app
EXPORT_DIR = "models"
os.makedirs(EXPORT_DIR, exist_ok=True)

# 1. Save your trained XGBoost model
joblib.dump(xgb_model, os.path.join(EXPORT_DIR, "xgb_model.pkl"))
print(" Saved trained XGBoost model to models/xgb_model.pkl")

# 2. Save model metadata and exact feature ordering
metadata = {
    "model_filename": "xgb_model.pkl",
    "algorithm": "XGBoost Regressor (Optimized)",
    "hyperparameters": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "random_state": 42
    },
    "metrics": {
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 2),
        "mse": round(float(mse), 2),
        "rmse": round(float(rmse), 2)
    },
    "feature_order": list(X_train.columns),
    "engineered_features": ["age_bmi", "smoker_bmi", "age_squared", "bmi_squared"],
    "currency": "USD",
    "currency_symbol": "$"
}

with open(os.path.join(EXPORT_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(" Saved metadata and feature order to models/model_metadata.json")
```

After running this cell, simply copy `models/xgb_model.pkl` and `models/model_metadata.json` into the `models/` directory of `insurance-prediction-app`.

---

## Quick Start Guide for macOS

### 1. Navigate to Project Directory
Open your macOS Terminal and navigate to the project directory:
```bash
cd /Users/ghoshankpawar/.gemini/antigravity/scratch/insurance-prediction-app
```

### 2. Create & Activate Virtual Environment
Using Python 3.12 (standard for Apple Silicon / Intel macOS):
```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Note for macOS OpenMP support: If XGBoost reports a missing `libomp.dylib`, running `brew install libomp` or copying scikit-learn's bundled `libomp.dylib` into the venv resolves it automatically).*

### 4. Model Artifact Setup
You have two simple options:
- **Option A (From Your Existing Notebook):** Copy your notebook's `xgb_model.pkl` and `model_metadata.json` into `models/`.
- **Option B (Train Locally with Exact Specifications):**
  ```bash
  python scripts/train_sample_model.py
  ```
  This trains the model using your exact hyperparameters (`n_estimators=300`, `learning_rate=0.05`, `max_depth=4`, `random_state=42`) and feature engineering, creating `models/xgb_model.pkl` and `models/model_metadata.json`.

### 5. Run Automated Tests
Execute the test suite to verify route responses, input validation, and model predictions:
```bash
pytest tests/test_prediction.py -v
```
All 9 test cases will run and confirm system integrity.

### 6. Start the Server

#### Development Mode (Default with auto-reload):
```bash
python app.py
```
*(Flask will show the informational notice `WARNING: This is a development server...`, which is standard during local development).*

#### Production WSGI Mode (Zero warnings, multi-threaded):
To run with a production WSGI server:
- **Using Waitress (recommended, simple):**
  ```bash
  python wsgi.py
  ```
- **Or Using Gunicorn:**
  ```bash
  gunicorn -w 2 -b 127.0.0.1:5001 wsgi:app
  ```
Output:
```text
2026-10-20 20:17:31 [INFO] insurepredict: Loaded model metadata successfully.
2026-10-20 20:17:31 [INFO] insurepredict: Successfully loaded XGBoost model from .../models/xgb_model.pkl
2026-10-20 20:17:31 [INFO] insurepredict: Starting InsurePredict AI server on http://127.0.0.1:5001
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5001
```

### 7. Open in Browser & Test Predictions
Open your browser (Safari, Chrome, or Firefox) and navigate to:
```
http://127.0.0.1:5001
```

1. Fill in **Age** (e.g. `35`).
2. Select **Biological Sex** (`Female` or `Male`).
3. Enter **BMI** (e.g. `27.4`) or use the built-in **"Need to calculate BMI?"** helper tool.
4. Enter **Dependents / Children** (e.g. `1`).
5. Choose **Smoking Status** (`Non-Smoker` or `Smoker`).
6. Select **Residential Region** (`Northeast`, `Northwest`, `Southeast`, or `Southwest`).
7. Click **"Predict Insurance Cost"**.
8. View the predicted annual cost, interaction feature breakdown, and Chart.js benchmark comparison!

### 8. Stopping the Server
In the terminal where Flask is running, press:
```bash
Ctrl + C
```
To deactivate the virtual environment when finished:
```bash
deactivate
```

---

## Configuring the Python Interpreter in Antigravity / VS Code

To ensure VS Code / Antigravity uses the virtual environment with all required packages:
1. Open the command palette: `Cmd + Shift + P`
2. Search and select: **Python: Select Interpreter**
3. Choose:
   ```text
   ./venv/bin/python
   ```
4. VS Code will now resolve all imports (`flask`, `xgboost`, `pandas`, `sklearn`, `pytest`) without warnings.

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Renders the responsive home page and interactive prediction calculator. |
| `GET` | `/health` | Returns server health status, model loading status, and timestamp in JSON. |
| `GET` | `/api/model-info` | Returns detailed model metadata, hyperparameters, and test evaluation metrics. |
| `POST` | `/predict` | Validates input data, performs feature engineering, and returns JSON prediction or rendered result page. |

### Sample Prediction Request
```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -H "X-Requested-With: XMLHttpRequest" \
  -d '{
    "age": 38,
    "sex": "male",
    "bmi": 28.5,
    "children": 2,
    "smoker": "no",
    "region": "northwest"
  }'
```

### Sample Prediction Response
```json
{
  "success": true,
  "prediction": 6842.15,
  "prediction_formatted": "$6,842.15",
  "currency": "USD",
  "currency_symbol": "$",
  "user_inputs": {
    "age": 38,
    "sex": "Male",
    "bmi": 28.5,
    "bmi_category": "Overweight",
    "bmi_risk": "Elevated risk",
    "children": 2,
    "smoker": "No",
    "region": "Northwest"
  },
  "engineered_features": {
    "age_bmi": 1083.0,
    "smoker_bmi": 0.0,
    "age_squared": 1444.0,
    "bmi_squared": 812.25
  },
  "insights": {
    "smoker_impact": "Low impact - non-smoker discount applied.",
    "bmi_impact": "Overweight (28.5) - Elevated risk",
    "model_used": "XGBoost Regressor (Optimized)"
  }
}
```

---

## Validation & Error Handling

- **Age:** Must be an integer between 1 and 120.
- **BMI:** Must be a float between 10.0 and 75.0. Negative values are strictly blocked.
- **Children:** Must be an integer between 0 and 20.
- **Smoking Status:** Must be one of `['yes', 'no']` (or 1/0).
- **Sex:** Must be one of `['male', 'female']`.
- **Region:** Must match dataset regions (`northeast`, `northwest`, `southeast`, `southwest`).
- **Missing Model Handling:** If `xgb_model.pkl` is absent, the backend returns HTTP 503 with helpful setup instructions rather than an unhandled crash or fake number.

---

## Medical & Financial Disclaimer

> **Notice:** This application provides an estimate based on a machine learning model and the data used to train it. Predictions are not actual insurance quotes, medical advice, or a guarantee of insurance eligibility or pricing.
