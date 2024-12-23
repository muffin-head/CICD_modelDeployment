# Wiki: Deploying a Machine Learning Model with ACR, AKS, and GitHub Actions

## Overview
This guide provides a step-by-step explanation of deploying a machine learning model using Azure Container Registry (ACR), Azure Kubernetes Service (AKS), and automating the process with GitHub Actions. The deployment includes building a Docker image, deploying the model as a microservice, and exposing it via a LoadBalancer.

---

## Prerequisites
- An active Azure subscription.
- Azure CLI installed for manual verification and debugging.
- Docker installed locally for building container images.
- GitHub repository to manage code and workflows.
- Terraform and GitHub Actions set up for IaC and CI/CD.

---

## Application Components
### **1. app.py**
This Python script uses Flask to expose a machine learning model as a REST API.

**Key Functionalities:**
- Loads a trained Random Forest model and a scaler from `.pkl` files.
- Accepts JSON input for predictions.
- Returns predictions or error messages in JSON format.

```python
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
        data = request.json
        input_df = pd.DataFrame(data)
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_df.fillna(0, inplace=True)
        scaled_features = scaler.transform(input_df)
        predictions = model.predict(scaled_features)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## Kubernetes Manifests
### **1. Deployment YAML**
Defines how the application is deployed in AKS, including the Docker image and environment variables.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
  labels:
    app: ml-model
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
        - name: ml-model
          image: acrterraformsepsisstreamingv1.azurecr.io/your-image-name:latest
          ports:
            - containerPort: 5000
          env:
            - name: ENV
              value: "production"
```

### **2. Service YAML**
Exposes the application to the internet via a LoadBalancer.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  type: LoadBalancer
  ports:
    - port: 80
      targetPort: 5000
  selector:
    app: ml-model
```

---

## Dockerfile
Defines the Docker image for the application.

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable to prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Run app.py when the container launches
CMD ["python", "app.py"]
```

---

## GitHub Actions Workflow
Automates the CI/CD pipeline to build, push, and deploy the application.

```yaml
name: Build, Push, and Deploy to AKS (v1)

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  build-push-deploy:
    name: Build, Push, and Deploy ML Model to AKS (v1)
    runs-on: ubuntu-latest

    steps:
      # Step 1: Checkout the repository
      - name: Checkout Code
        uses: actions/checkout@v3

      # Step 2: Authenticate with Azure CLI using Device Code
      - name: Authenticate with Azure CLI
        run: az login --use-device-code

      # Step 3: Set Azure Subscription
      - name: Set Azure Subscription
        run: az account set --subscription "1fad5e2e-cd4b-437b-8fff-ee503b4e0e29"

      # Step 4: Log in to Azure Container Registry (v1)
      - name: Log in to Azure Container Registry
        run: az acr login --name acrterraformSepsisStreamingv1

      # Step 5: Build and Push Docker Image
      - name: Build and Push Docker Image
        run: |
          docker build -t acrterraformsepsisstreamingv1.azurecr.io/your-image-name:latest .
          docker push acrterraformsepsisstreamingv1.azurecr.io/your-image-name:latest

      # Step 6: Get AKS Credentials (v1)
      - name: Configure kubectl for AKS
        run: |
          az aks get-credentials \
            --resource-group rg-aks-acr-terraformv1 \
            --name aks-terraform-clusterv1 \
            --overwrite-existing

      # Step 7: Apply Kubernetes Manifests
      - name: Deploy to AKS
        run: |
          kubectl apply -f k8s/deployment.yaml
          kubectl apply -f k8s/service.yaml

      # Step 8: Redeploy with Updated Image
      - name: Redeploy ML Model with Updated Image
        run: |
          kubectl set image deployment/ml-model-deployment \
            ml-model=acrterraformsepsisstreamingv1.azurecr.io/your-image-name:latest

      # Step 9: Wait for Resources to Stabilize
      - name: Wait for Resources to Stabilize
        run: |
          echo "Waiting for pods and service to stabilize..."
          sleep 30

      # Step 10: Verify Deployment
      - name: Check Pods and External IP
        run: |
          kubectl get pods
          kubectl get service ml-model-service
```

---

## Local Testing
### **Testing the Model Locally**
Use the `model.py` script to test the model and scaler locally:

```python
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

model_path = os.path.join('RandomForest', 'model.pkl')
scaler_path = os.path.join('scaler', 'scaler.pkl')

# Load the trained model and scaler
with open(model_path, 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Sample data
sample_data = pd.DataFrame({
    'MDC_PULS_OXIM_PULS_RATE_Result_min': [80, 75],
    'MDC_TEMP_Result_mean': [36.5, 37.0],
    'MDC_PULS_OXIM_PULS_RATE_Result_mean': [85, 78],
    'HR_to_RR_Ratio': [1.2, 1.1]
})

# Preprocess the data
sample_data.replace([np.inf, -np.inf], np.nan, inplace=True)
sample_data.fillna(0, inplace=True)
sample_data_scaled = scaler.transform(sample_data)

# Make predictions
predictions = rf_model.predict(sample_data_scaled)
print('Predictions:', predictions)
```

---

## Deployment Workflow
1. **Push Code to GitHub:**
   - Include `app.py`, `Dockerfile`, `k8s/deployment.yaml`, and `k8s/service.yaml`.

2. **Trigger GitHub Actions Workflow:**
   - Push to the `main` branch or manually trigger the workflow.

3. **Monitor the Workflow:**
   - Check the GitHub Actions logs for the status of each step.

4. **Verify Deployment in AKS:**
   - Run `kubectl get pods` and `kubectl get service ml-model-service` to confirm the deployment.

5. **Test the Application:**
   - Use `curl` or a similar tool to send a POST request:
     ```bash
     curl -X POST -H "Content-Type: application/json" --data '[
       {
         "MDC_PULS_OXIM_PULS_RATE_Result_min": 80,
         "MDC_TEMP_Result_mean": 36.5,
         "MDC_PULS_OXIM_PULS_RATE_Result_mean": 85,
         "HR_to_RR_Ratio": 1.2
       }
     ]' http://<EXTERNAL-IP>/predict
     ```

---

## Future Enhancements
1. **Integrate Azure Key Vault:**
   - Securely store sensitive information like ACR credentials and AKS kubeconfig.

2. **Enable Autoscaling in AKS:**
   - Configure horizontal and vertical pod autoscalers.

3. **Advanced Monitoring:**
   - Use Azure Monitor and Application Insights for detailed metrics and logging.

---

## Summary
This guide provides a comprehensive approach to deploying a machine learning model with ACR, AKS, and GitHub Actions. By following this workflow, you ensure a robust, scalable, and automated deployment pipeline for your ML applications.

