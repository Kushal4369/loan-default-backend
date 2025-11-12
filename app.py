from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("./../loan_default_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    age = data["age"]
    income = data["income"]
    loan_amount = data["loan_amount"]
    credit_history = data["credit_history"]
    gender = 1 if data["gender"] == "Male" else 0
    married = 1 if data["married"] == "Yes" else 0

    features = np.array([[age, income, loan_amount, gender, married, credit_history]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    result = {
        "prediction": int(prediction),
        "probability": float(probability)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
