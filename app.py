from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model once at startup
model = joblib.load("Models/linear_regression_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    input_data = {
    "gender": request.form["gender"],
    "age": float(request.form["age"]),
    "hypertension": int(request.form["hypertension"]),
    "heart_disease": int(request.form["heart_disease"]),
    "smoking_history": request.form["smoking_history"],
    "bmi": float(request.form["bmi"]),
    "HbA1c_level": float(request.form["HbA1c_level"]),
    "blood_glucose_level": float(request.form["blood_glucose_level"]),
    }

    input_df = pd.DataFrame([input_data])
    preprocessor = joblib.load("Pipelines/preprocessor.pkl")
    features_array = preprocessor.transform(input_df)

    prediction = model.predict(features_array)

    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    print(prediction)
    return render_template("index.html", prediction_text=f'The patient is {result} with {features_array} and {prediction}.')

if __name__ == "__main__":
    app.run(debug=True)