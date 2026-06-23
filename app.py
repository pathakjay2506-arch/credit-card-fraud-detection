from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model, scaler and metrics
with open("model/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

FEATURE_NAMES = [
    "transaction_amount", "time_of_day", "merchant_category",
    "distance_from_home", "foreign_transaction", "online_transaction",
    "transaction_frequency", "account_age_days",
    "prev_fraud_flag", "device_trust_score"
]

@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = [float(data[f]) for f in FEATURE_NAMES]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        confidence = round(float(max(probability)) * 100, 2)

        return jsonify({
            "prediction": int(prediction),
            "label": "FRAUDULENT" if prediction == 1 else "LEGITIMATE",
            "confidence": confidence,
            "fraud_prob": round(float(probability[1]) * 100, 2),
            "legit_prob": round(float(probability[0]) * 100, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)