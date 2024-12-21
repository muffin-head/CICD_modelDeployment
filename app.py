from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
from azure.eventhub import EventHubConsumerClient

# Initialize Flask App
app = Flask(__name__)

# Load the model and scaler from MLflow
model_path = "./model/RandomForest"  # Ensure this matches the location where your model is stored
scaler_path = "./model/scaler"
model = mlflow.sklearn.load_model(model_path)
scaler = mlflow.sklearn.load_model(scaler_path)

# Define required feature columns
REQUIRED_FEATURES = [
    "MDC_PULS_OXIM_PULS_RATE_Result_min",
    "MDC_TEMP_Result_mean",
    "MDC_PULS_OXIM_PULS_RATE_Result_mean",
    "HR_to_RR_Ratio"
]

# Event Hub connection details
EVENT_HUB_CONNECTION_STR = "Endpoint=sb://sepsisstreamingeventhubnamespace.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=HmtoeA1c8SpIls4m6VV55l79cIj/+AIAa+AEhPX1xDA="
EVENT_HUB_NAME = "eventhubsepsisstreaminge"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert JSON to Pandas DataFrame
        input_df = pd.DataFrame(input_data)

        # Check for missing features
        missing_features = [feature for feature in REQUIRED_FEATURES if feature not in input_df.columns]
        if missing_features:
            return jsonify({"error": f"Missing required features: {', '.join(missing_features)}"}), 400

        # Preprocess data using StandardScaler
        scaled_data = scaler.transform(input_df)

        # Predict using the loaded model
        predictions = model.predict(scaled_data)

        # Return predictions
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/eventhub-stream', methods=['POST'])
def stream_from_eventhub():
    """
    Stream data from Event Hub, preprocess it, and make predictions.
    """
    try:
        def process_event(event_data):
            # Parse the incoming data
            input_data = pd.DataFrame([event_data])
            
            # Validate required features
            missing_features = [feature for feature in REQUIRED_FEATURES if feature not in input_data.columns]
            if missing_features:
                print(f"Missing features in event: {missing_features}")
                return
            
            # Preprocess the data
            scaled_data = scaler.transform(input_data)

            # Predict using the loaded model
            predictions = model.predict(scaled_data)
            print(f"Predictions: {predictions.tolist()}")
        
        # Initialize Event Hub Consumer
        client = EventHubConsumerClient.from_connection_string(
            conn_str=EVENT_HUB_CONNECTION_STR,
            consumer_group="$Default",
            eventhub_name=EVENT_HUB_NAME
        )

        # Process messages from Event Hub
        with client:
            client.receive(
                on_event=process_event,
                starting_position="-1"  # Start from the beginning of the stream
            )

        return jsonify({"status": "Streaming and predicting from Event Hub"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "Healthy"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
