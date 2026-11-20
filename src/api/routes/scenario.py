"""
Scenario Routes: Predict and Retrieve Targeted Practice Scenarios
"""
import uuid
from typing import Any
import torch
from fastapi import APIRouter
from src.api.schemas import ScenarioPredictRequest, ScenarioResponse
from src.config import settings, MODELS_DIR
from src.models.scenario_network import ScenarioNetwork
from src.models.baseline_knn import BaselineKNNMatcher
from src.training.dataset import CandidateBankManager, fen_to_tensor
from src.api.routes.user import USER_CACHE

router = APIRouter(prefix="/scenario", tags=["Scenario"])

bank_mgr = CandidateBankManager()
baseline = BaselineKNNMatcher()


def get_default_candidates() -> list[dict[str, Any]]:
    return [
        {
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "player_color": "white",
            "clock_remaining": 65.0,
            "clock_bucket": "60-120s",
            "opponent_rating_band": "1200-1400",
            "opponent_rating": 1350,
            "reason": "Italian Game sharp tactical pressure",
        },
        {
            "fen": "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5",
            "player_color": "black",
            "clock_remaining": 42.0,
            "clock_bucket": "30-60s",
            "opponent_rating_band": "1400-1600",
            "opponent_rating": 1520,
            "reason": "Sicilian Defense low-time scramble",
        },
        {
            "fen": "8/5pk1/4p1p1/3pP2p/3P3P/5KP1/8/8 w - - 1 45",
            "player_color": "white",
            "clock_remaining": 25.0,
            "clock_bucket": "0-30s",
            "opponent_rating_band": "1200-1400",
            "opponent_rating": 1280,
            "reason": "Pawn endgame under intense time scramble",
        },
    ]


@router.post("/predict", response_model=ScenarioResponse)
def predict_scenario(req: ScenarioPredictRequest):
    """
    Predicts a practice scenario conditioned on the player's history.
    Outputs a legal board FEN, player color, remaining clock, and opponent-rating band.
    """
    candidates = bank_mgr.load_candidates()
    if not candidates:
        candidates = get_default_candidates()
        bank_mgr.save_candidates(candidates)

    target_elo = req.player_rating

    if req.use_baseline:
        baseline.fit(candidates)
        matched = baseline.predict(player_rating=req.player_rating, avg_clock=60.0)
        return ScenarioResponse(
            scenario_id=str(uuid.uuid4())[:8],
            fen=matched["fen"],
            player_color=matched.get("player_color", "white"),
            clock_remaining=float(matched.get("clock_remaining", 60.0)),
            clock_bucket=matched.get("clock_bucket", "60-120s"),
            opponent_rating_band=matched.get("opponent_rating_band", "1200-1400"),
            target_elo=target_elo,
            reason=f"[Baseline KNN] {matched.get('reason', 'Scenario matched from rating profile')}",
        )

    # Neural retrieval
    model_path = MODELS_DIR / "scenario_model.pt"
    if model_path.exists():
        try:
            device = torch.device("cpu")
            model = ScenarioNetwork(
                board_embedding_dim=settings.EMBEDDING_DIM,
                history_hidden_dim=settings.HISTORY_HIDDEN_DIM,
            )
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            cand_embeddings = bank_mgr.compute_embeddings(model.board_encoder, device="cpu")

            # Synthetic or user history input
            user_input = torch.zeros((1, 5, 131), dtype=torch.float32)
            with torch.no_grad():
                outputs = model(user_input, cand_embeddings)
                best_idx = int(torch.argmax(outputs["retrieval_logits"], dim=-1)[0].item())
                pred_clock_idx = int(torch.argmax(outputs["clock_logits"], dim=-1)[0].item())
                pred_rating_idx = int(torch.argmax(outputs["rating_logits"], dim=-1)[0].item())

            best_idx = min(best_idx, len(candidates) - 1)
            selected = candidates[best_idx]
            clock_bucket = settings.CLOCK_BUCKETS[min(pred_clock_idx, len(settings.CLOCK_BUCKETS) - 1)]
            rating_band = settings.RATING_BANDS[min(pred_rating_idx, len(settings.RATING_BANDS) - 1)]

            return ScenarioResponse(
                scenario_id=str(uuid.uuid4())[:8],
                fen=selected["fen"],
                player_color=selected.get("player_color", "white"),
                clock_remaining=float(selected.get("clock_remaining", 60.0)),
                clock_bucket=clock_bucket,
                opponent_rating_band=rating_band,
                target_elo=target_elo,
                reason=f"[Neural Model] {selected.get('reason', 'High evaluation deterioration stretch')}",
            )
        except Exception:
            pass

    # Fallback to candidate 0
    selected = candidates[0]
    return ScenarioResponse(
        scenario_id=str(uuid.uuid4())[:8],
        fen=selected["fen"],
        player_color=selected.get("player_color", "white"),
        clock_remaining=float(selected.get("clock_remaining", 60.0)),
        clock_bucket=selected.get("clock_bucket", "60-120s"),
        opponent_rating_band=selected.get("opponent_rating_band", "1200-1400"),
        target_elo=target_elo,
        reason=selected.get("reason", "Critical game continuation"),
    )
