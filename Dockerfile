# Summarizer Flask app: model from Azure Blob Storage (AZURE_MODEL_BLOB_URL) or MLflow registry (MLFLOW_MODEL_URI) if blob not set
FROM python:3.11-slim

WORKDIR /app

# Install dependencies (CPU-only PyTorch to keep image small; default torch is ~10GB+ with CUDA)
COPY requirements-app.txt requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir flask "python-dotenv>=1.0.0" gunicorn "mlflow>=2.0" "transformers>=4.56,<5" accelerate

# App code (no model in image; model loaded at runtime from Azure Blob or MLflow)
COPY app.py .
COPY templates/ templates/

# Model dir: app downloads from Azure Blob when AZURE_MODEL_BLOB_URL is set, else uses MLflow registry
RUN mkdir -p /app/model

EXPOSE 5001

# Single worker (model in memory); threads for light concurrency
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
