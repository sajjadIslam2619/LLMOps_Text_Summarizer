# LLMOps Text Summarizer

A Flask web app for **text summarization** using a fine-tuned **Flan-T5** model. The project covers training with MLflow, exporting the model, and deploying via Docker and Azure App Service. The model can be loaded from **Azure Blob Storage** at runtime (recommended for production) or from an MLflow registry.

---

## Summary of the codebase

- **`app.py`** – Flask app that serves a simple UI and a `/summarize` endpoint. It loads the Flan-T5 pipeline from either an Azure Blob URL (zip) or an MLflow model URI, then runs inference with configurable `MAX_NEW_TOKENS`.
- **`summarization_llm_MLflow_experiments.ipynb`** – Jupyter notebook for training/evaluating Flan-T5 summarization, logging runs and metrics (e.g. ROUGE-L) to MLflow, and optionally uploading the chosen model to Azure Blob Storage.
- **`templates/`** – HTML template for the summarizer UI.
- **`Dockerfile`** – Builds a Linux image that runs the app with Gunicorn. The image does **not** bundle the model; the app downloads it from Azure Blob at startup when `AZURE_MODEL_BLOB_URL` is set.
- **`requirements-app.txt`** – Python dependencies for the Flask app (Flask, MLflow, transformers, PyTorch CPU-only in Docker, etc.).
- **`DEPLOY_STEPS.txt`** – End-to-end steps: train in the notebook, pick the best run in MLflow, export and zip the model, upload to Azure Blob, build the Docker image, push to Docker Hub, and run or deploy the container.
- **`DEPLOY_AZURE.txt`** – Commands to deploy the Docker image from Docker Hub to **Azure App Service** (resource group, plan, web app, port 5001, and `AZURE_MODEL_BLOB_URL`).

---

## Prerequisites

- **Python 3.11+** (for local run and notebook)
- **Docker** (for building and running the container)
- **Azure CLI** (optional; for deploying to Azure App Service)
- An **Azure Storage** account and container (for storing `model.zip`), and optionally an MLflow tracking server

---

## Project structure (main items)

```
.
├── app.py                    # Flask summarizer app
├── templates/
│   └── index.html            # Web UI
├── summarization_llm_MLflow_experiments.ipynb   # Train & log with MLflow
├── requirements-app.txt      # App dependencies
├── Dockerfile                # Container build (no model in image)
├── .env.example              # Example env vars (copy to .env)
├── DEPLOY_STEPS.txt          # Full deployment flow (notebook → Docker Hub)
├── DEPLOY_AZURE.txt          # Azure App Service deployment commands
└── README.md                 # This file
```

---

## Instructions

### 1. Clone and setup (local)

```bash
git clone <your-repo-url>
cd LLMOps_Text_Summarizer
```

Create a virtual environment and install app dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements-app.txt
```

Copy the example env file and set your model URL (see [Environment variables](#environment-variables)):

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/macOS
# Edit .env and set AZURE_MODEL_BLOB_URL (or MLFLOW_MODEL_URI for local MLflow).
```

### 2. Run the app locally

```bash
python app.py
```

Open **http://localhost:5001**, paste text, and use “Summarize”. The first request may be slow while the model is downloaded (from Azure Blob) or loaded from MLflow.

### 3. Build and run with Docker

**Pre-built image (Docker Hub):** [sajjadislam1926/flan-t5-text-summarizer-app](https://hub.docker.com/repository/docker/sajjadislam1926/flan-t5-text-summarizer-app/general)

From the project root:

```bash
docker build -t text-summarizer-app:1.0 .
docker run -p 5001:5001 -e AZURE_MODEL_BLOB_URL="https://<your-storage>.blob.core.windows.net/<container>/model.zip?<sas>" text-summarizer-app:1.0
```

Then open **http://localhost:5001**. See **DEPLOY_STEPS.txt** for the full flow (including pushing to Docker Hub).

### 4. Deploy to Azure App Service

Use the commands in **DEPLOY_AZURE.txt** to create a resource group, App Service plan, and Web App using your Docker Hub image, and set `WEBSITES_PORT=5001` and `AZURE_MODEL_BLOB_URL`. The app URL will be `https://<app-name>.azurewebsites.net`.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_MODEL_BLOB_URL` | Yes (for production) | Full URL to `model.zip` in Azure Blob (include SAS token if private). |
| `MLFLOW_MODEL_URI` | No | MLflow model URI when not using Azure Blob (e.g. `models:/Flan-T5-Summarization@champion`). |
| `MODEL_CACHE_DIR` | No | Directory for cached model (default: `model` locally, `/app/model` in Docker). |
| `MAX_NEW_TOKENS` | No | Max tokens for the summary (default: `80`). |

Do not commit `.env` or secrets; use `.env.example` as a template.

---

## Training and model export

1. Open **`summarization_llm_MLflow_experiments.ipynb`** and run the cells to train and log runs to MLflow.
2. In the MLflow UI, pick the best run (e.g. by ROUGE-L).
3. Export that run (or a registered model) to a folder, zip it, and upload the zip to Azure Blob Storage.
4. Use the blob URL (with SAS if needed) as `AZURE_MODEL_BLOB_URL` when running or deploying the app.

Detailed steps are in **DEPLOY_STEPS.txt**.

---

## Helpful tutorials

- **MLflow:** [YouTube tutorial](https://www.youtube.com/watch?v=6ngxBkx05Fs)
- **Docker:** [YouTube tutorial](https://www.youtube.com/watch?v=pg19Z8LL06w)

---

## License

See the repository license file if present.
