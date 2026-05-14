

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime

# Import ML components (same as your training code)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ================================

# Initialize Flask App

# ================================

app = Flask(**name**)

# ================================

# LOAD PRE-TRAINED MODEL

# ================================

# NOTE: In production, you would load a saved model like:

# best_model = joblib.load(‘best_churn_model.pkl’)

# 

# For now, we’ll create a dummy model for demonstration

# Replace this with your actual trained model

def load_model():
“””
Load your trained model here.
If you’ve saved it: joblib.load(‘path_to_model.pkl’)
“””
try:
# Uncomment when you have saved model
# model = joblib.load(‘best_churn_model.pkl’)
# return model

```
    # FOR NOW: Create pipeline structure (you train it with your data)
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                       'PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    model = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(max_iter=1000, C=1))
    ])
    
    return model
    
except Exception as e:
    print(f"Error loading model: {e}")
    return None
```

# Load model on startup

best_model = load_model()

# ================================

# HELPER FUNCTIONS

# ================================

def validate_input(data):
“””
Validate JSON input data
Returns: (is_valid, error_message)
“””
required_fields = [‘tenure’, ‘MonthlyCharges’, ‘TotalCharges’, ‘Contract’]

```
for field in required_fields:
    if field not in data:
        return False, f"Missing required field: {field}"

# Validate data types
try:
    float(data['MonthlyCharges'])
    float(data['TotalCharges'])
    int(data['tenure'])
except ValueError:
    return False, "Invalid data types. Numbers expected."

return True, None
```

def prepare_input(raw_data):
“””
Convert raw JSON to DataFrame for model prediction
“””
# Create a single-row DataFrame
df = pd.DataFrame([raw_data])

```
# Fill missing columns with default values
# (In production, handle this more carefully)
return df
```

# ================================

# API ENDPOINTS

# ================================

@app.route(’/’, methods=[‘GET’])
def home():
“”“Health check endpoint”””
return jsonify({
“status”: “ok”,
“message”: “Churn Prediction API is running”,
“version”: “1.0”,
“author”: “Umer Qureshi”
}), 200

@app.route(’/predict’, methods=[‘POST’])
def predict():
“””
Main prediction endpoint

```
Expected JSON input:
{
    "tenure": 12,
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
    "gender": "Male",
    "Contract": "Month-to-month",
    ... (other features)
}

Returns:
{
    "prediction": "Yes" or "No",
    "churn_probability": 0.75,
    "confidence": "High/Medium/Low",
    "timestamp": "2025-01-15T10:30:00"
}
"""

try:
    # Get JSON data
    data = request.get_json()
    
    if not data:
        return jsonify({
            "error": "No JSON data provided"
        }), 400
    
    # Validate input
    is_valid, error_msg = validate_input(data)
    if not is_valid:
        return jsonify({
            "error": error_msg
        }), 400
    
    # Prepare data for model
    X = prepare_input(data)
    
    # Make prediction
    prediction = best_model.predict(X)[0]  # 0 or 1
    prediction_proba = best_model.predict_proba(X)[0][1]  # Probability of churn
    
    # Convert to readable format
    churn_label = "Yes" if prediction == 1 else "No"
    
    # Determine confidence level
    if prediction_proba > 0.7:
        confidence = "High"
    elif prediction_proba > 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Prepare response
    response = {
        "status": "success",
        "prediction": churn_label,
        "churn_probability": round(prediction_proba, 4),
        "confidence": confidence,
        "timestamp": datetime.now().isoformat(),
        "risk_level": "HIGH" if prediction_proba > 0.7 else "MEDIUM" if prediction_proba > 0.4 else "LOW"
    }
    
    return jsonify(response), 200

except Exception as e:
    return jsonify({
        "error": str(e)
    }), 500
```

@app.route(’/batch-predict’, methods=[‘POST’])
def batch_predict():
“””
Batch prediction endpoint

```
Expected JSON input:
{
    "customers": [
        {"tenure": 12, "MonthlyCharges": 65.5, ...},
        {"tenure": 24, "MonthlyCharges": 85.0, ...}
    ]
}
"""

try:
    data = request.get_json()
    
    if 'customers' not in data:
        return jsonify({"error": "Missing 'customers' key"}), 400
    
    customers = data['customers']
    
    if not isinstance(customers, list):
        return jsonify({"error": "'customers' must be a list"}), 400
    
    results = []
    
    for i, customer in enumerate(customers):
        try:
            is_valid, error_msg = validate_input(customer)
            if not is_valid:
                results.append({
                    "customer_id": i,
                    "error": error_msg
                })
                continue
            
            X = prepare_input(customer)
            prediction = best_model.predict(X)[0]
            prediction_proba = best_model.predict_proba(X)[0][1]
            
            churn_label = "Yes" if prediction == 1 else "No"
            
            results.append({
                "customer_id": i,
                "prediction": churn_label,
                "churn_probability": round(prediction_proba, 4)
            })
        
        except Exception as e:
            results.append({
                "customer_id": i,
                "error": str(e)
            })
    
    return jsonify({
        "status": "success",
        "total_processed": len(customers),
        "results": results
    }), 200

except Exception as e:
    return jsonify({"error": str(e)}), 500
```

@app.route(’/info’, methods=[‘GET’])
def info():
“”“Get API information and expected input format”””
return jsonify({
“api_name”: “Churn Prediction API”,
“version”: “1.0”,
“author”: “Umer Qureshi”,
“endpoints”: {
“GET /”: “Health check”,
“POST /predict”: “Single prediction”,
“POST /batch-predict”: “Batch predictions”,
“GET /info”: “This endpoint”
},
“example_input”: {
“tenure”: 12,
“MonthlyCharges”: 65.5,
“TotalCharges”: 786.0,
“gender”: “Male”,
“SeniorCitizen”: 0,
“Partner”: “No”,
“Dependents”: “No”,
“PhoneService”: “Yes”,
“Contract”: “Month-to-month”,
“InternetService”: “Fiber optic”
}
}), 200

# ================================

# ERROR HANDLERS

# ================================

@app.errorhandler(404)
def not_found(error):
return jsonify({“error”: “Endpoint not found”}), 404

@app.errorhandler(500)
def server_error(error):
return jsonify({“error”: “Internal server error”}), 500

# ================================

# RUN SERVER

# ================================

if **name** == ‘**main**’:
print(”=” * 50)
print(“🚀 CHURN PREDICTION API STARTING”)
print(”=” * 50)
print(“Author: Umer Qureshi”)
print(“Model: Logistic Regression + SMOTE”)
print(”=” * 50)
print(”\n📊 Available Endpoints:”)
print(”  GET  http://localhost:5000/”)
print(”  POST http://localhost:5000/predict”)
print(”  POST http://localhost:5000/batch-predict”)
print(”  GET  http://localhost:5000/info”)
print(”\n” + “=” * 50)

```
# Run on localhost:5000
app.run(debug=True, host='127.0.0.1', port=5000)
```