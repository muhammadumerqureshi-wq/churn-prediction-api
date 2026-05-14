from flask import Flask, jsonify, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load('churn_model.pkl')

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "ok", "message": "Churn API running"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Make prediction using actual model
    prediction = model.predict([list(data.values())])[0]
    probability = model.predict_proba([list(data.values())])[0][1]
    
    return jsonify({
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "probability": float(probability),
        "confidence": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
    })

if __name__ == '__main__':
    app.run(debug=False, port=5000)
