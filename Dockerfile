FROM continuumio/miniconda3:4.12.0

# Set working directory
WORKDIR /app

# Copy conda environment file and requirements.txt
COPY conda.yaml .
COPY requirements.txt .

# Create conda environment and install dependencies
RUN conda env create -f conda.yaml && \
    conda init bash && \
    echo "source activate mlflow-env" > ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc && conda install pip && pip install -r requirements.txt"

# Copy the application code
COPY . .

# Expose the port used by the Flask app
EXPOSE 5000

# Run the application
CMD ["/bin/bash", "-c", "source activate mlflow-env && python app.py"]
