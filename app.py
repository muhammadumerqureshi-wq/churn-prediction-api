from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "ok", "message": "API running"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    return jsonify({
        "prediction": "Yes",
        "probability": 0.75,
        "confidence": "High"
    })

if __name__ == '__main__':
    app.run(debug=False, port=5000)
