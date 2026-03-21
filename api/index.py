from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Correct paths (relative)
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "linear_regression_model.pkl"))
preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))

@app.route("/")
def home():
    return "API is working"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_data = {
        "gender": data["gender"],
        "age": float(data["age"]),
        "hypertension": int(data["hypertension"]),
        "heart_disease": int(data["heart_disease"]),
        "smoking_history": data["smoking_history"],
        "bmi": float(data["bmi"]),
        "HbA1c_level": float(data["HbA1c_level"]),
        "blood_glucose_level": float(data["blood_glucose_level"]),
    }

    input_df = pd.DataFrame([input_data])
    features_array = preprocessor.transform(input_df)
    prediction = model.predict(features_array)

    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

    return jsonify({
        "prediction": result
    })