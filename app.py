from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import asyncio
from azure.eventhub.aio import EventHubProducerClient
from azure.eventhub import EventData
import json
from datetime import datetime
from azure.eventhub import TransportType

app = Flask(__name__)

# Load the scaler
with open('scaler/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the model
with open('RandomForest/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Azure Event Hub configuration
EVENT_HUB_CONNECTION_STRING = "Endpoint=sb://sepsisstreamingeventhubnamespace.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=HmtoeA1c8SpIls4m6VV55l79cIj/+AIAa+AEhPX1xDA="
EVENT_HUB_NAME = "eventhubsepsisstreaming"

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

        # Prepare data for streaming
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_data": data,
            "predictions": predictions.tolist()
        }
        result_json = json.dumps(result)

        # Send data to Event Hub
        asyncio.run(send_to_eventhub(result_json))

        return jsonify({'predictions': predictions.tolist(), 'message': "Sent to Event Hub"})
    except Exception as e:
        return jsonify({'error': str(e)})


async def send_to_eventhub(data):
    """Send prediction data to Azure Event Hub."""
    producer = EventHubProducerClient.from_connection_string(
        conn_str=EVENT_HUB_CONNECTION_STRING,
        eventhub_name=EVENT_HUB_NAME,
        transport_type=TransportType.AmqpOverWebsocket
    )

    async with producer:
        # Create a batch and add data
        event_data_batch = await producer.create_batch()
        event_data_batch.add(EventData(data))

        # Send the batch
        await producer.send_batch(event_data_batch)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
