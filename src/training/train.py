"""
Training Pipeline for Scenario Prediction Network
Logs metrics and parameters, saves checkpoints.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.config import settings, MODELS_DIR, USER_MODEL_DIR
from src.models.scenario_network import ScenarioNetwork
from src.training.dataset import HistoryScenarioDataset, CandidateBankManager

logger = logging.getLogger(__name__)


def train_scenario_model(
    train_samples: list[dict[str, Any]],
    candidate_bank: list[dict[str, Any]],
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    save_path: Optional[Path] = None,
    bank_mgr: Optional[CandidateBankManager] = None,
    init_from: Optional[Path] = None,
    persist_bank: bool = False,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Trains multi-task scenario network.

    ``bank_mgr`` controls which bank the candidate embeddings are read from and
    written to. Callers for named users pass a private manager so their data
    never overwrites the global demo bank. ``init_from`` optionally fine-tunes
    an existing checkpoint instead of starting from random weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bank_mgr = bank_mgr or CandidateBankManager()
    if persist_bank:
        bank_mgr.save_candidates(candidate_bank)
    else:
        bank_mgr.candidates = candidate_bank

    model = ScenarioNetwork(
        board_embedding_dim=settings.EMBEDDING_DIM,
        history_hidden_dim=settings.HISTORY_HIDDEN_DIM,
        num_clock_buckets=len(settings.CLOCK_BUCKETS),
        num_rating_bands=len(settings.RATING_BANDS),
    )

    if init_from is not None and Path(init_from).exists():
        checkpoint = torch.load(init_from, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Fine-tuning from checkpoint %s", init_from)

    model = model.to(device)

    # Precompute candidate embeddings
    candidate_embeddings = bank_mgr.compute_embeddings(model.board_encoder, device=str(device))
    candidate_embeddings = candidate_embeddings.to(device)

    dataset = HistoryScenarioDataset(train_samples, board_encoder=model.board_encoder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history_loss = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            history = batch["history"].to(device)
            target_cand = batch["target_candidate_idx"].to(device)
            target_clock = batch["target_clock_bucket"].to(device)
            target_rating = batch["target_rating_band"].to(device)

            optimizer.zero_grad()
            outputs = model(history, candidate_embeddings)

            loss_retrieval = criterion_ce(outputs["retrieval_logits"], target_cand)
            loss_clock = criterion_ce(outputs["clock_logits"], target_clock)
            loss_rating = criterion_ce(outputs["rating_logits"], target_rating)

            total_loss = loss_retrieval + 0.5 * loss_clock + 0.5 * loss_rating
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / max(1, len(dataloader))
        history_loss.append(avg_loss)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Save model
    out_path = save_path or (MODELS_DIR / "scenario_model.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epochs": epochs,
        "final_loss": history_loss[-1] if history_loss else 0.0,
        "candidate_count": len(candidate_bank),
        "bank_md5": hashlib.md5(
            json.dumps(candidate_bank, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    if metadata:
        checkpoint.update(metadata)
    torch.save(checkpoint, out_path)

    return {
        "status": "success",
        "final_loss": history_loss[-1] if history_loss else 0.0,
        "model_path": str(out_path),
    }


def train_personalized_model(
    username: str,
    bank: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Fine-tune the Lichess-pretrained base model on one player's Chess.com pairs.

    The global checkpoint is the base model trained on Lichess games, so a
    personalized model never depends on the player's data alone. Returns
    ``None`` when there is not enough data, so serving falls back to the base
    model while still restricting retrieval to this player's bank.
    """
    if len(pairs) < 2 or len(bank) < 2:
        return None

    bank_mgr = CandidateBankManager.for_user(username)
    global_checkpoint = MODELS_DIR / "scenario_model.pt"
    out_path = USER_MODEL_DIR / f"{username}.pt"

    result = train_scenario_model(
        pairs,
        bank,
        epochs=5,
        batch_size=min(8, max(1, len(pairs))),
        lr=1e-3,
        save_path=out_path,
        bank_mgr=bank_mgr,
        init_from=global_checkpoint if global_checkpoint.exists() else None,
    )
    result["owner"] = username
    result["personalized"] = True
    return result


if __name__ == "__main__":
    """DVC train stage: real chronologically-paired samples when data is available."""
    logging.basicConfig(level=logging.INFO)

    from src.config import RAW_DATA_DIR
    from src.data.chess_com_client import load_params
    from src.training.dataset import build_training_pairs_from_bank

    params = load_params()
    train_params = params.get("train", {})
    username = str(params.get("data", {}).get("username", "bonc_1")).lower()

    bank_mgr = CandidateBankManager()
    candidates = bank_mgr.load_candidates()

    # Seed defaults if the extraction stage has not produced a bank yet.
    if not candidates:
        from src.api.routes.scenario import get_default_candidates

        candidates = get_default_candidates()
        bank_mgr.save_candidates(candidates)

    raw_games: list[dict[str, Any]] = []
    raw_path = RAW_DATA_DIR / f"{username}_games.json"
    if raw_path.exists():
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_games = json.load(f)
    raw_games = raw_games[: int(params.get("extract", {}).get("max_games", 30))]

    samples: list[dict[str, Any]] = []
    if raw_games and len(candidates) > 1:
        samples = build_training_pairs_from_bank(raw_games, username, candidates)
        logger.info("Built %d real training pairs from the extracted bank", len(samples))

    if not samples:
        logger.warning("No real pairs available; falling back to synthetic smoke data")
        samples = [
            {
                "history": [
                    {
                        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                        "clock_remaining": 590.0,
                        "player_rating": 1300,
                        "opponent_rating": 1350,
                        "ply": 2,
                    }
                ],
                "target_candidate_idx": 0,
                "target_clock_bucket": "60-120s",
                "target_rating_band": "1200-1400",
            }
        ]

    result = train_scenario_model(
        samples,
        candidates,
        epochs=int(train_params.get("epochs", 5)),
        batch_size=int(train_params.get("batch_size", 8)),
        lr=float(train_params.get("lr", 1e-3)),
    )
    print("Training result:", result)
