import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR.parent.joinpath("uploaded_images")
UPLOAD_DIR.mkdir(exist_ok=True)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "PBL")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./Med-Images")

# Model cache
_model_cache = {
    "model": None,
    "tokenizer": None,
    "model_name": None
}
