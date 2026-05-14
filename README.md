# churn-prediction-api


# Churn Prediction ML API

Production-ready Flask API for customer churn prediction.

## Features
- Single & batch predictions
- Input validation
- Error handling
- JSON responses

## Endpoints
- GET / - Health check
- POST /predict - Single prediction
- POST /batch-predict - Batch predictions
- GET /info - API info

## Setup
pip install -r requirements.txt
python churn_flask_api.py

## Usage
curl -X POST http://localhost:5000/predict \
  -d '{"tenure": 12, "MonthlyCharges": 65.5, ...}'
