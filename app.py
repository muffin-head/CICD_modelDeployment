from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the scaler
with open('scaler/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the model
with open('RandomForest/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        data = request.json
        # Convert data to DataFrame
        input_df = pd.DataFrame(data)
        # Preprocess input data
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_df.fillna(0, inplace=True)
        # Scale input features
        scaled_features = scaler.transform(input_df)
        # Make predictions
        predictions = model.predict(scaled_features)
        # Return predictions as JSON
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
