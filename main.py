#Disclaimer: This file only to be used for referencing purposes

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = "modelannMLP.pkl"  # Ensure this path matches your saved model
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from form
        input_features = [
            "Female", "Male", "Govt_job", "Never_worked", "Private", "Self-employed",
            "children", "Unknown", "formerly_smoked", "never_smoked", "smokes", "Rural",
            "Urban", "age", "hypertension", "heart_disease", "ever_married",
            "avg_glucose_level", "bmi"
        ]
        data = [float(request.form.get(feature)) for feature in input_features]
        features = np.array(data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        prediction_text = "Stroke Risk: High" if prediction[0] == 1 else "Stroke Risk: Low"

        return render_template("index.html", prediction_text=prediction_text)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

#Note: Please ensure the correct file path in your local machine and also as required change the port at your end. 