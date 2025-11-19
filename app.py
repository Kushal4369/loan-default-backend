from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # allow frontend (Next.js) to access backend
scaler = joblib.load("scaler.pkl")
model = joblib.load("loan_default_model.pkl")


@app.route('/')
def home():
    return jsonify({"message": "Flask ML API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # --- Extract input values from JSON ---
        Age = float(data.get("Age", 0))
        Income = float(data.get("Income", 0))
        Family = int(data.get("Family", 1))
        CCAvg = float(data.get("CCAvg", 0))
        Education = int(data.get("Education", 1))
        Mortgage = float(data.get("Mortgage", 0))
        Securities_Account = int(data.get("Securities_Account", False))
        CD_Account = int(data.get("CD_Account", False))
        Online = int(data.get("Online", False))
        CreditCard = int(data.get("CreditCard", False))

        features = np.array([[Age, Income, Family, CCAvg, Education,
                              Mortgage, Securities_Account,
                              CD_Account, Online, CreditCard]])

        prediction = model.predict(scaler.transform(features))
        prediction = bool(prediction[0])  # Convert numpy type to native Python boolean  

        # Construct response
        result = {"default": prediction}

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Run the app
    app.run(host="0.0.0.0", port=5000, debug=True)
