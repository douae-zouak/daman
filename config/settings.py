"""
config/settings.py — Configuration centrale du projet
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Force la désactivation du parallélisme
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

# ─── LLM ─────────────────────────────────────────────────────────────────────
LLM_PROVIDER    = "mistral"
LLM_MODEL       = "mistral-small"
LLM_API_KEY     = os.getenv("MISTRAL_API_KEY")
LLM_TEMPERATURE = 0.2

# ─── EMBEDDINGS ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# ─── CHEMINS ─────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(__file__))
TXT_DIR       = os.path.join(BASE_DIR, "data", "documents")
FAQ_JSON_PATH = os.path.join(BASE_DIR, "data", "faq", "frequent_questions.json")
CHROMA_DIR    = os.path.join(BASE_DIR, "chroma_db")
UPLOADS_DIR   = os.path.join(BASE_DIR, "data", "uploads")

# ─── CHUNKING ────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80

# ─── RETRIEVAL ───────────────────────────────────────────────────────────────
TOP_K_DOCS          = 3
TOP_K_FAQ           = 2
FAQ_SCORE_THRESHOLD = 0.75

# ─── BASE DE DONNÉES ─────────────────────────────────────────────────────────
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'data', 'app.db')}"

# ─── AUTHENTIFICATION GOOGLE OAUTH ───────────────────────────────────────────
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# ─── JWT SESSION ─────────────────────────────────────────────────────────────
SECRET_KEY      = os.getenv("SECRET_KEY", "change-this-secret-key-in-production")
JWT_ALGORITHM   = "HS256"
JWT_EXPIRE_DAYS = 7

# ─── EMAIL (Gmail SMTP) ───────────────────────────────────────────────────────
SMTP_HOST     = "smtp.gmail.com"
SMTP_PORT     = 587
SMTP_USER     = os.getenv("SMTP_USER")      # ex: votre.adresse@gmail.com
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")  # mot de passe d'application Gmail

# ─── APPLICATION ─────────────────────────────────────────────────────────────
APP_URL = os.getenv("APP_URL", "http://127.0.0.1:8000")