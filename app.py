from flask import Flask, request, jsonify, Response
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
import pickle
import numpy as np
import pandas as pd
import time

app = Flask(__name__)

# Load the scaler
with open('scaler/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the model
with open('RandomForest/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prometheus metric for real-time predictions
current_prediction = Gauge('current_prediction', 'Real-time prediction value', ['timestamp'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        data = request.json
        input_df = pd.DataFrame(data)
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_df.fillna(0, inplace=True)
        scaled_features = scaler.transform(input_df)
        predictions = model.predict(scaled_features)

        # Update Prometheus metrics
        for prediction in predictions:
            current_time = int(time.time())  # Current timestamp
            current_prediction.labels(timestamp=str(current_time)).set(prediction)

        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

# Prometheus metrics endpoint
@app.route('/metrics', methods=['GET'])
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
