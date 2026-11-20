"""
Pawnder (PressureLab) - Configuration & Constants
"""
import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CANDIDATE_BANK_DIR = DATA_DIR / "candidate_bank"
MODELS_DIR = BASE_DIR / "models_saved"

for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CANDIDATE_BANK_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    PROJECT_NAME: str = "Pawnder - PressureLab Chess Practice"
    API_PREFIX: str = "/api"
    DEFAULT_TIME_CONTROL: str = "600"  # 10+0 rapid (600 seconds)
    CHESS_COM_USER_AGENT: str = "PawnderMLOps/1.0 (contact: admin@pawnder.local)"
    
    # Engine Settings
    STOCKFISH_PATH: str = os.getenv("STOCKFISH_PATH", "")
    DEFAULT_ENGINE_DEPTH: int = 12
    DEFAULT_BLUNDER_THRESHOLD_CP: int = 150
    DEFAULT_WIN_PROB_DROP_THRESHOLD: float = 0.20
    
    # Model parameters
    BOARD_CHANNELS: int = 14  # 6 white pieces, 6 black pieces, active color, castling/rights
    EMBEDDING_DIM: int = 128
    HISTORY_HIDDEN_DIM: int = 256
    CLOCK_BUCKETS: list[str] = ["0-30s", "30-60s", "60-120s", "120s+"]
    RATING_BANDS: list[str] = ["<1000", "1000-1200", "1200-1400", "1400-1600", "1600-1800", "1800+"]


settings = Settings()
