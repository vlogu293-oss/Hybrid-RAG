import os
from pathlib import Path
from dotenv import load_dotenv


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# Explicitly load the project's root .env
ENV_FILE = PROJECT_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)


# --------------------------------------------------
# DATA / STORAGE
# --------------------------------------------------

DATA_DIR = BASE_DIR / "data"

VECTOR_PATH = BASE_DIR / os.getenv(
    "VECTOR_PATH",
    "vector_store"
)

EVAL_DIR = BASE_DIR / "evals"

PDF_PATH = DATA_DIR / "spotify_web_app_architecture.pdf"


# --------------------------------------------------
# OPENAI
# --------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)


# --------------------------------------------------
# NEO4J
# --------------------------------------------------

NEO4J_URI = os.getenv("NEO4J_URI")

NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")

NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

NEO4J_DATABASE = os.getenv(
    "NEO4J_DATABASE",
    "neo4j"
)


# --------------------------------------------------
# RAG SETTINGS
# --------------------------------------------------

TOP_K_VECTOR = int(
    os.getenv("TOP_K_VECTOR", "5")
)

TOP_K_GRAPH = int(
    os.getenv("TOP_K_GRAPH", "5")
)


# --------------------------------------------------
# CREATE DIRECTORIES
# --------------------------------------------------

VECTOR_PATH.mkdir(
    parents=True,
    exist_ok=True
)

EVAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)