# Use the official Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy model directories (RandomForest and Scaler) to the container
COPY ./RandomForest /app/RandomForest
COPY ./scaler /app/scaler

# Copy environment and requirements files (assumes both folders have the same dependencies)
COPY ./RandomForest/conda.yaml /app/
COPY ./RandomForest/requirements.txt /app/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip==21.2.4 setuptools==61.2.0 wheel==0.37.0
RUN pip install -r requirements.txt

# Expose port for the Flask application
EXPOSE 5000

# Set default command to run Flask app (update this if the app script is different)
CMD ["python", "app.py"]
