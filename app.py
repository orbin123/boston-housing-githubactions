from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("models/housing_model.pkl")


@app.route("/")
def home():
    return "Boston Housing Prediction API is running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Expecting input as a list of 13 features
    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)

    return jsonify({
        "prediction": f'${float(prediction[0] * 1000):.2f}'
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)