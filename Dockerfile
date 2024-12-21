FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the entire RandomForest directory
COPY ./RandomForest /app/RandomForest

# Set up Python dependencies
RUN pip install -r /app/RandomForest/requirements.txt

# Copy application code
COPY app.py /app/

# Expose the application port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
