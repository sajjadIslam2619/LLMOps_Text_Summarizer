# Summarizer Flask app (model loaded from Azure Blob at runtime)
FROM python:3.11-slim

WORKDIR /app

# Install dependencies (CPU-only PyTorch to keep image small; default torch is ~10GB+ with CUDA)
COPY requirements-app.txt requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir flask "python-dotenv>=1.0.0" gunicorn "mlflow>=2.0" "transformers>=4.56,<5" accelerate

# App code (no model in image; set AZURE_MODEL_BLOB_URL when running)
COPY app.py .
COPY templates/ templates/

# Model is downloaded from Azure Blob at startup via AZURE_MODEL_BLOB_URL
RUN mkdir -p /app/model

EXPOSE 5001

# Single worker (model in memory); threads for light concurrency
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
