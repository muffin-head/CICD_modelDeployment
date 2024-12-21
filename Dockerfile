FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the model folder, including RandomForest and Scaler
COPY model /app/model

# Install dependencies for both RandomForest and Scaler
RUN pip install -r /app/model/RandomForest/requirements.txt
RUN pip install -r /app/model/Scaler/requirements.txt

# Copy the application file
COPY app.py /app/

# Expose the application port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
