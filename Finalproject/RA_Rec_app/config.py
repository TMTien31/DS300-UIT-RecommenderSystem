"""
Configuration file for RA-Rec Chatbot
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
NOTEBOOKS_DIR = BASE_DIR.parent / "notebooks" / "RA_Rec"

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model configurations
MODEL_NAME = "models/gemini-2.5-flash-lite"
SBERT_MODEL = "keepitreal/vietnamese-sbert"

# File paths
RECIPES_CSV = DATA_DIR / "all_recipes_final.csv"
EMBEDDINGS_FILE = NOTEBOOKS_DIR / "recipes_embeddings_list.pkl"
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
