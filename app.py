import os
import pandas as pd
import numpy as np
import joblib
import time
from flask import Flask, request, jsonify
from prometheus_client import Counter,Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)

MODEL_DIR = "models"
CURRENT_MODEL_FILE = os.path.join(MODEL_DIR, "current_model.txt")

# Read active model name
with open(CURRENT_MODEL_FILE, "r") as f:
    model_name = f.read().strip()

model_path = os.path.join(MODEL_DIR, model_name)

# Load model
model = joblib.load(model_path)

# Metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API Requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency"
)

MODEL_MSE = Gauge(
    "model_mse",
    "Mean Squared Error of current model"
)

MODEL_MAE = Gauge(
    "model_mae",
    "Mean Absolute Error of current model"
)

MODEL_VERSION = Gauge(
    "model_version",
    "Version number of current model"
)

# Evaluate model on test data
X_test = pd.read_csv("data/test/X_test.csv")
y_test = pd.read_csv("data/test/y_test.csv").values.ravel()

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

MODEL_MSE.set(mse)
MODEL_MAE.set(mae)

# Extract version number from model name
import re
match = re.search(r"model_v(\d+)\.joblib", model_name)
if match:
    MODEL_VERSION.set(int(match.group(1)))

print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")


@app.route("/")
def home():
    return "Boston Housing API Running"


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()

    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)

        response = jsonify({
            "prediction": float(prediction[0])
        })
        status = 200

    except Exception as e:
        response = jsonify({"error": str(e)})
        status = 500

    REQUEST_COUNT.labels("POST", "/predict", status).inc()
    REQUEST_LATENCY.observe(time.time() - start)

    return response, status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)