from flask import Flask, request, jsonify
import joblib
import requests
import pandas as pd
import io
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

MODEL_URL = "https://huggingface.co/xenoxu/Diabetes_Prediction_Model/resolve/main/linear_regression_model.pkl"
PREPROCESSOR_URL = "https://huggingface.co/xenoxu/Diabetes_Prediction_Model/resolve/main/preprocessor.pkl"

def load_joblib_from_url(url):
    response = requests.get(url)
    return joblib.load(io.BytesIO(response.content))

# Load once
model = load_joblib_from_url(MODEL_URL)
preprocessor = load_joblib_from_url(PREPROCESSOR_URL)

@app.route("/")
def home():
    return "API running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data received"}), 400

        input_df = pd.DataFrame([data])

        features = preprocessor.transform(input_df)
        prediction = model.predict(features)

        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})
    
# if __name__ == "__main__":
#     app.run(debug=True)