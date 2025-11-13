from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # allow frontend (Next.js) to access backend
scaler = joblib.load("scaler.pkl")
model = joblib.load("loan_default_model.pkl")


# ------------------------------
# (Optional) Load your ML model
# Uncomment and replace with your real model file
# model = joblib.load("model.pkl")
# ------------------------------

@app.route('/')
def home():
    return jsonify({"message": "Flask ML API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # --- Extract input values from JSON ---
        ID = float(data.get("ID", 0))
        Age = float(data.get("Age", 0))
        Experience = float(data.get("Experience", 0))
        Income = float(data.get("Income", 0))
        ZIP_Code = float(data.get("ZIP_Code", 0))
        Family = int(data.get("Family", 1))
        CCAvg = float(data.get("CCAvg", 0))
        Education = int(data.get("Education", 1))
        Mortgage = float(data.get("Mortgage", 0))
        Personal_Loan = int(data.get("Personal_Loan", False))
        Securities_Account = int(data.get("Securities_Account", False))
        CD_Account = int(data.get("CD_Account", False))
        Online = int(data.get("Online", False))
        CreditCard = int(data.get("CreditCard", False))

        features = np.array([[
            ID, Age, Experience, Income, ZIP_Code, Family, CCAvg, Education,
            Mortgage, Personal_Loan, Securities_Account, CD_Account, Online, CreditCard
        ]])

        prediction = model.predict(scaler.transform(features))[0]

        # Construct response
        result = {"default": prediction}

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Run the app
    app.run(host="0.0.0.0", port=5000, debug=True)
