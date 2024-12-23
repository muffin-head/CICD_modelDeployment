from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
import json
from datetime import datetime

app = Flask(__name__)

# Load the scaler
with open('scaler/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the model
with open('RandomForest/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Azure Data Lake Storage connection string
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=datalaketfexample;AccountKey=7bZ7+qULn3sdaGQXqfJzzohWqqy172fopVsPA7X341sr31rdSUnUqPQrIN3aPz9Xi/U9Z/2Z/alu+AStkU42pg==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "gold"

# Initialize Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        data = request.json
        input_df = pd.DataFrame(data)

        # Preprocess input data
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_df.fillna(0, inplace=True)

        # Scale input features
        scaled_features = scaler.transform(input_df)

        # Make predictions
        predictions = model.predict(scaled_features)

        # Prepare data for saving
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_data": data,
            "predictions": predictions.tolist()
        }
        result_json = json.dumps(result)

        # Save to Azure Data Lake
        blob_name = f"predictions_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
        container_client.upload_blob(name=blob_name, data=result_json, overwrite=True)

        return jsonify({'predictions': predictions.tolist(), 'message': f"Saved to ADLS as {blob_name}"})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
