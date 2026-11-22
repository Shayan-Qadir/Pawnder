"""
DVC stage: pretrain the global scenario network on the Lichess BASE dataset.

This checkpoint (models_saved/scenario_model.pt) is the starting point for
every personalized model. Chess.com player data is used only afterwards, as a
fine-tuning step, so the network is never trained on user data alone.

Every run is logged to MLflow with its params, dataset versions and metrics;
the resulting checkpoint is registered as a versioned model artifact.

Input : data/raw/lichess_base_games.json + data/candidate_bank/lichess_base.json
Output: models_saved/scenario_model.pt
"""
import json
import logging
import hashlib
from pathlib import Path
from typing import Any

from src.config import RAW_DATA_DIR, CANDIDATE_BANK_DIR, MODELS_DIR
from src.data.chess_com_client import load_params
from src.data.validation import validate_games, validate_candidates, ValidationError
from src.tracking import start_run, log_params, log_metrics, log_artifact, register_model
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


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    params = load_params()
    train_params = params.get("train", {})

    raw_path = RAW_DATA_DIR / BASE_GAMES_FILE
    bank_path = CANDIDATE_BANK_DIR / BASE_BANK_FILE

    # Fail before training if the base datasets are missing, malformed or illegal.
    try:
        raw_games = validate_games(raw_path)
        candidates = validate_candidates(bank_path)
    except ValidationError as exc:
        raise SystemExit(f"Base dataset validation failed: {exc}") from exc

    samples = build_training_pairs_from_bank(raw_games, None, candidates)
    logger.info("Built %d chronological pairs from the Lichess base dataset", len(samples))
    if not samples:
        raise SystemExit("No Lichess base training pairs were produced.")

    # Explicit bank keeps embeddings separate from the Chess.com demo bank.
    bank_mgr = CandidateBankManager(bank_path=bank_path)
    with start_run("train_base_lichess", tags={"stage": "train_base", "dataset": "lichess"}) as _:
        log_params({
            "epochs": train_params.get("epochs", 5),
            "batch_size": train_params.get("batch_size", 8),
            "lr": train_params.get("lr", 1e-3),
            "embedding_dim": train_params.get("embedding_dim", 128),
            "hidden_dim": train_params.get("hidden_dim", 256),
            "dataset": "lichess",
            "games_md5": _md5(raw_path),
            "bank_md5": _md5(bank_path),
            "train_pairs": len(samples),
        })
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
        log_metrics({
            "final_loss": result["final_loss"],
            "train_pairs": len(samples),
            "bank_size": len(candidates),
        })
        log_artifact(MODELS_DIR / "scenario_model.pt")
        version = register_model(MODELS_DIR / "scenario_model.pt")
        if version:
            result["registered_model_version"] = version

    result["base_dataset"] = "lichess"
    print("Base training result:", result)
