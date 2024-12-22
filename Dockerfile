FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy model and application code
COPY ./model/RandomForest /app/model/RandomForest
COPY ./model/scaler /app/model/scaler
COPY requirements.txt /app/
COPY ./model/RandomForest/requirements.txt /app/RandomForest-requirements.txt
COPY ./model/scaler/requirements.txt /app/scaler-requirements.txt

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r /app/RandomForest-requirements.txt && \
    pip install -r /app/scaler-requirements.txt

# Copy the application code
COPY app.py /app/

# Expose the application port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
