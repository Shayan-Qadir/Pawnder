"""
Training Pipeline for Scenario Prediction Network
Logs metrics and parameters, saves checkpoints.
"""
import logging
from pathlib import Path
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.config import settings, MODELS_DIR
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
) -> dict[str, Any]:
    """Trains multi-task scenario network."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bank_mgr = CandidateBankManager()
    bank_mgr.save_candidates(candidate_bank)

    model = ScenarioNetwork(
        board_embedding_dim=settings.EMBEDDING_DIM,
        history_hidden_dim=settings.HISTORY_HIDDEN_DIM,
        num_clock_buckets=len(settings.CLOCK_BUCKETS),
        num_rating_bands=len(settings.RATING_BANDS),
    ).to(device)

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
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": epochs,
            "final_loss": history_loss[-1] if history_loss else 0.0,
        },
        out_path,
    )

    return {
        "status": "success",
        "final_loss": history_loss[-1] if history_loss else 0.0,
        "model_path": str(out_path),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing training module execution...")
    # Synthetic mini-dataset for smoke verification
    candidates = [
        {
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "player_color": "white",
            "clock_remaining": 80.0,
            "clock_bucket": "60-120s",
            "opponent_rating_band": "1200-1400",
        },
        {
            "fen": "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5",
            "player_color": "black",
            "clock_remaining": 40.0,
            "clock_bucket": "30-60s",
            "opponent_rating_band": "1400-1600",
        },
    ]

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

    res = train_scenario_model(samples, candidates, epochs=1, batch_size=1)
    print("Smoke training result:", res)
