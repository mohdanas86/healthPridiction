# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pickle
# import numpy as np

# # Load the trained model
# with open("health_model.pkl", "rb") as f:
#     model = pickle.load(f)

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Define a prediction route
# @app.route("/predict", methods=["POST"])
# def predict():
#     # Parse incoming JSON data
#     data = request.json
#     try:
#         # Extract features and convert them to numeric
#         features = np.array([
#             float(data["age"]),
#             float(data["bmi"]),
#             float(data["glucose"])
#         ]).reshape(1, -1)

#         # Make prediction
#         prediction = model.predict(features)[0]
#         # risk = "High" if prediction == 1 else if prediction === 0.5  else "Low"
#         if(prediction == 1):
#             risk = "High"
#         elif (prediction == 0.5):
#             risk = "Medium"
#         else:
#             risk = "Low"

#         return jsonify({"risk_level": risk})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400


# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Load the trained model
with open("health_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define a prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Parse incoming JSON data
    data = request.json
    try:
        # Extract features and convert them to numeric
        features = np.array([
            float(data["age"]),
            float(data["bmi"]),
            float(data["glucose"])
        ]).reshape(1, -1)

        # Get the probability of the positive class (diabetes)
        probability = model.predict_proba(features)[0][1]

        # Determine risk level based on probability thresholds
        if probability > 0.7:
            risk = "High"
        elif 0.4 < probability <= 0.7:
            risk = "Medium"
        else:
            risk = "Low"

        return jsonify({"risk_level": risk, "probability": probability})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
