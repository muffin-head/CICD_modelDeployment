import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Define paths to the model and scaler files
model_path = os.path.join('RandomForest', 'model.pkl')
scaler_path = os.path.join('scaler', 'scaler.pkl')

# Load the trained Random Forest model
with open(model_path, 'rb') as model_file:
    rf_model = pickle.load(model_file)

# Load the Standard Scaler
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Sample data for testing (replace with your actual test data)
# Ensure the feature order matches the training data
sample_data = pd.DataFrame({
    'MDC_PULS_OXIM_PULS_RATE_Result_min': [80, 75],
    'MDC_TEMP_Result_mean': [36.5, 37.0],
    'MDC_PULS_OXIM_PULS_RATE_Result_mean': [85, 78],
    'HR_to_RR_Ratio': [1.2, 1.1]
})

# Preprocess the sample data
# Handle missing and infinite values
sample_data.replace([np.inf, -np.inf], np.nan, inplace=True)
sample_data.fillna(0, inplace=True)

# Scale the features using the loaded scaler
sample_data_scaled = scaler.transform(sample_data)

# Make predictions using the loaded model
predictions = rf_model.predict(sample_data_scaled)

# If you have true labels for the sample data, calculate the F1 score
# Replace 'true_labels' with your actual labels
true_labels = [1, 0]  # Example true labels
f1 = f1_score(true_labels, predictions)
print(f'F1 Score: {f1}')

# Output the predictions
print('Predictions:', predictions)
