from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)

try:
    model = joblib.load(os.path.join(BASE_DIR, "linear_regression_model.pkl"))
    preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading error:", str(e))

@app.route("/")
def home():
    return "API is working"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("Incoming data:", data)

        input_df = pd.DataFrame([data])
        features_array = preprocessor.transform(input_df)
        prediction = model.predict(features_array)

        return jsonify({"prediction": str(prediction[0])})

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({"error": str(e)})