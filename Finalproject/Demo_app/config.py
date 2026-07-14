import os
from pathlib import Path

from dotenv import load_dotenv


def normalize_openai_compatible_base_url(raw_base_url: str | None) -> str | None:
    """Normalize LiteLLM/OpenAI-compatible base URLs for the OpenAI SDK."""
    if not raw_base_url:
        return None
    base_url = raw_base_url.strip().rstrip("/")
    if not base_url:
        return None
    if not base_url.startswith(("http://", "https://")):
        base_url = f"https://{base_url}"
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


# Base paths
BASE_DIR = Path(__file__).resolve().parent
FINALPROJECT_ROOT = BASE_DIR.parent
PROJECT_ROOT = FINALPROJECT_ROOT.parent
DATA_DIR = FINALPROJECT_ROOT / "data"
NOTEBOOKS_DIR = FINALPROJECT_ROOT / "notebooks"
SAVED_MODELS_DIR = NOTEBOOKS_DIR / "Saved_models"

# Load repository-level defaults first, then allow Demo_app/.env to override them.
load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(BASE_DIR / ".env", override=True)

# OpenAI-compatible LLM configuration, including LiteLLM proxies.
LLM_BASE_URL = normalize_openai_compatible_base_url(os.getenv("LLM_BASE_URL"))
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "local-std-03")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "300"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))

# Frontend/API request timeout. Chat can be slower than search because it calls the LLM.
API_REQUEST_TIMEOUT_SECONDS = float(os.getenv("API_REQUEST_TIMEOUT_SECONDS", "360"))

# Model configurations
SBERT_MODEL = "keepitreal/vietnamese-sbert"

# File paths
RECIPES_CSV = DATA_DIR / "all_recipes_final.csv"
RAREC_EMBEDDINGS_FILE = SAVED_MODELS_DIR / "RA_Rec" / "recipes_embeddings_list.pkl"
LEGACY_RAREC_EMBEDDINGS_FILE = NOTEBOOKS_DIR / "RA_Rec" / "recipes_embeddings_list.pkl"
EMBEDDINGS_FILE = RAREC_EMBEDDINGS_FILE if RAREC_EMBEDDINGS_FILE.exists() else LEGACY_RAREC_EMBEDDINGS_FILE
STATE_FILE = BASE_DIR / "state.json"

# Search parameters
TOP_K_RESULTS = 10
TOP_K_DISPLAY = 5

# Default state structure
DEFAULT_STATE = {
    "hard_constraints": {
        "type_of_food": [],
        "ingredients": []
    },
    "soft_constraints": {
        "cook_time": [],
        "num_of_people": [],
        "calories": [],
        "algeric": []
    },
    "recommended_items": [],
    "accepted_items": [],
    "rejected_items": []
}

# Greeting message
GREETING_MESSAGE = """Xin chào! Tôi là trợ lý gợi ý món ăn, tên là Tiến Bịp. 
Bạn muốn tìm món ăn gì? (ví dụ: món Tết, món nhanh, món cho 4 người...)"""

# Restart keywords
RESTART_KEYWORDS = [
    "bắt đầu lại", "bat dau lai", "restart", "reset",
    "làm lại", "lam lai", "start over", "bắt đầu từ đầu"
]
