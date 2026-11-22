"""
DVC stage: pretrain the global scenario network on the Lichess BASE dataset.

This checkpoint (models_saved/scenario_model.pt) is the starting point for
every personalized model. Chess.com player data is used only afterwards, as a
fine-tuning step, so the network is never trained on user data alone.

Input : data/raw/lichess_base_games.json + data/candidate_bank/lichess_base.json
Output: models_saved/scenario_model.pt
"""
import json
import logging
from pathlib import Path
from typing import Any

from src.config import RAW_DATA_DIR, CANDIDATE_BANK_DIR, MODELS_DIR
from src.data.chess_com_client import load_params
from src.training.dataset import CandidateBankManager, build_training_pairs_from_bank
from src.training.train import train_scenario_model

logger = logging.getLogger(__name__)

BASE_GAMES_FILE = "lichess_base_games.json"
BASE_BANK_FILE = "lichess_base.json"


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing base artifact: {path}. Run the earlier base stages first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    params = load_params()
    train_params = params.get("train", {})

    raw_games = load_json(RAW_DATA_DIR / BASE_GAMES_FILE)
    candidates = load_json(CANDIDATE_BANK_DIR / BASE_BANK_FILE)

    if len(candidates) < 2:
        raise SystemExit("Lichess base bank is too small to pretrain; check the extract_base stage.")

    samples = build_training_pairs_from_bank(raw_games, None, candidates)
    logger.info("Built %d chronological pairs from the Lichess base dataset", len(samples))

    if not samples:
        raise SystemExit("No Lichess base training pairs were produced.")

    # Explicit bank keeps embeddings separate from the Chess.com demo bank.
    bank_mgr = CandidateBankManager(bank_path=CANDIDATE_BANK_DIR / BASE_BANK_FILE)
    result = train_scenario_model(
        samples,
        candidates,
        epochs=int(train_params.get("epochs", 5)),
        batch_size=int(train_params.get("batch_size", 8)),
        lr=float(train_params.get("lr", 1e-3)),
        save_path=MODELS_DIR / "scenario_model.pt",
        bank_mgr=bank_mgr,
        metadata={
            "base_dataset": "lichess",
            "base_games": len(raw_games),
            "base_bank_entries": len(candidates),
        },
    )
    result["base_dataset"] = "lichess"
    print("Base training result:", result)
