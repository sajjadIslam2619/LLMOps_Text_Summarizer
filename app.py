"""
Flask app for text summarization using the Flan-T5 model (loaded from MLflow).
Uses model from Azure Blob Storage when AZURE_MODEL_BLOB_URL is set; otherwise MLFLOW_MODEL_URI.
"""
import os

# Load .env so AZURE_MODEL_BLOB_URL can be set in a file (copy from .env.example)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import zipfile
import tempfile
import urllib.request
from pathlib import Path
import mlflow.transformers

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Model source: Azure Blob URL (zip) takes precedence; else MLFLOW_MODEL_URI
AZURE_MODEL_BLOB_URL = os.environ.get("AZURE_MODEL_BLOB_URL")

MODEL_URI = os.environ.get(
    "MLFLOW_MODEL_URI",
    "models:/Flan-T5-Summarization@champion" if not AZURE_MODEL_BLOB_URL else None,
)
# Use a path that works on Windows (local) and Linux (Docker); avoid /app/model on Windows
_LOCAL_MODEL_DIR = os.environ.get("MODEL_CACHE_DIR")
if _LOCAL_MODEL_DIR is None:
    _LOCAL_MODEL_DIR = "/app/model" if os.path.isdir("/app") else os.path.join(os.getcwd(), "model")
LOCAL_MODEL_DIR = _LOCAL_MODEL_DIR
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "80"))
PREFIX = "Summarize the following text:\n\n"

pipeline = None


def ensure_model_from_blob(blob_url: str, target_dir: str):
    """
    Download the model only if it does not already exist locally.
    """
    target_path = Path(target_dir)

    if (target_path / "MLmodel").exists():
        print("Using cached model.", flush=True)
        return str(target_path)

    print("Downloading model from Azure Blob...", flush=True)

    target_path.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        req = urllib.request.Request(blob_url, headers={"User-Agent": "SummarizerApp"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            with open(tmp_path, "wb") as f:
                f.write(resp.read())

        with zipfile.ZipFile(tmp_path, "r") as zf:
            # Skip "." and ".." and other unsafe names (avoids "No such file: \\app\model\." on Windows)
            for name in zf.namelist():
                if name in (".", "..") or name.startswith("/") or ".." in name:
                    continue
                zf.extract(name, target_dir)

        print("Model downloaded and extracted.", flush=True)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return str(target_path)

def get_model():
    global pipeline

    if pipeline is None:

        if AZURE_MODEL_BLOB_URL:
            model_path = ensure_model_from_blob(
                AZURE_MODEL_BLOB_URL,
                LOCAL_MODEL_DIR
            )
            # file:// URI: must be absolute (resolve() fixes "relative path can't be expressed as a file URI")
            uri = Path(model_path).resolve().as_uri()
            pipeline = mlflow.transformers.load_model(uri)

        elif MODEL_URI:
            pipeline = mlflow.transformers.load_model(MODEL_URI)

    return pipeline


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    text = (request.form.get("text") or (request.json or {}).get("text", "")).strip()
    if not text:
        return jsonify({"error": "Please enter some text to summarize."}), 400
    try:
        pipe = get_model()
        prompt = f"{PREFIX}{text}"
        out = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        summary = (out[0]["generated_text"] or "").strip()
        return jsonify({"summary": summary})
    except Exception as e:
        # Avoid leaking stack traces/paths to clients in production
        msg = str(e) if os.environ.get("FLASK_DEBUG") else "An error occurred while summarizing."
        return jsonify({"error": msg}), 500


if __name__ == "__main__":
    # Debug on when running locally (python app.py); Docker uses gunicorn and does not use this block
    app.run(debug=True, host="0.0.0.0", port=5001)
