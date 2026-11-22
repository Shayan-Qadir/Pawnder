"""
Scenario Routes: Predict and Retrieve Targeted Practice Scenarios
"""
import hashlib
import json
import random
import re
import uuid
from typing import Any, Optional

import chess
import torch
from fastapi import APIRouter, HTTPException

from src.api.schemas import ScenarioPredictRequest, ScenarioResponse
from src.config import settings, MODELS_DIR, USER_MODEL_DIR
from src.models.scenario_network import ScenarioNetwork
from src.models.baseline_knn import BaselineKNNMatcher
from src.training.dataset import CandidateBankManager, encode_history_tensor
from src.api.routes.user import USER_CACHE, _load_profile_from_disk

router = APIRouter(prefix="/scenario", tags=["Scenario"])

bank_mgr = CandidateBankManager()
baseline = BaselineKNNMatcher()

# Cache candidate embeddings per (model, bank) version. Recomputing a CNN pass
# over the whole bank on every request was the dominant serving cost.
_EMBEDDING_CACHE: dict[str, torch.Tensor] = {}

# Recently served FENs per user. Each "Generate New Scenario" click must give
# the player a genuinely different position, not the deterministic argmax one.
_RECENT_SCENARIOS: dict[str, list[str]] = {}
_MAX_RECENT = 8


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
            "player_color": "white",
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


def _bank_version(candidates: list[dict[str, Any]]) -> str:
    return hashlib.md5(
        json.dumps(candidates, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _profile_for(username: str) -> Optional[dict[str, Any]]:
    return USER_CACHE.get(username) or _load_profile_from_disk(username)


def _history_tensor(
    req: ScenarioPredictRequest, model: ScenarioNetwork, username: Optional[str]
) -> torch.Tensor:
    """Use the importing player's real blunder history; zeros only for guests."""
    profile = _profile_for(username) if username else None
    history = profile.get("sample_blunders", []) if profile else []
    if not history:
        return torch.zeros((1, 5, settings.EMBEDDING_DIM + 3), dtype=torch.float32)

    rating = float(profile.get("player_rating", 1300.0))
    enriched = []
    for blunder in history:
        step = dict(blunder)
        step.setdefault("player_rating", rating)
        enriched.append(step)
    return encode_history_tensor(enriched, model.board_encoder, max_seq_len=5)


def _align_color(fen: str, player_color: str) -> tuple[str, str]:
    """Ensure the side to move in the FEN matches the requested player color."""
    board = chess.Board(fen)
    if board.turn == chess.WHITE and player_color != "white":
        return board.mirror().fen(), "black"
    if board.turn == chess.BLACK and player_color != "black":
        return board.mirror().fen(), "white"
    return fen, player_color


def _load_bank(username: str, manager: CandidateBankManager) -> list[dict[str, Any]]:
    """A named user can only ever see scenarios extracted from their own games."""
    candidates = [
        dict(c)
        for c in manager.load_candidates()
        if c.get("owner", username) == username
    ]
    if not candidates:
        raise HTTPException(
            status_code=422,
            detail="No personal scenarios found. Import this player first.",
        )
    return candidates


def _band_midpoint(band: str) -> int:
    match = re.match(r"(\d+)-(\d+)", band or "")
    if match:
        return (int(match.group(1)) + int(match.group(2))) // 2
    if "1800+" in (band or ""):
        return 1900
    if "<1000" in (band or ""):
        return 900
    return 1400


def _scenario_target_elo(
    candidate: dict[str, Any], player_rating: int, clock_bucket: str, rating_band: str
) -> int:
    """Derive Stockfish strength from the drill itself, not a fixed 1400 default.

    The opponent from the player's own game is the main signal (the exact rating
    when available, otherwise the band midpoint); the clock pressure bucket
    adjusts it so a frantic time-scramble drill is a little more forgiving. The
    player's own rating keeps the level relevant to them.
    """
    opponent_rating = candidate.get("opponent_rating")
    band_elo = (
        float(opponent_rating)
        if opponent_rating
        else _band_midpoint(rating_band or candidate.get("opponent_rating_band", ""))
    )
    clock_adj = {
        "0-30s": -150,
        "30-60s": -75,
        "60-120s": 0,
        "120s+": 50,
    }.get(clock_bucket, 0)
    raw = 0.7 * band_elo + 0.3 * player_rating + clock_adj
    return int(max(1000, min(2400, round(raw / 50) * 50)))


def _remember_fen(username: str, fen: str) -> None:
    recent = _RECENT_SCENARIOS.setdefault(username, [])
    recent.append(fen)
    _RECENT_SCENARIOS[username] = recent[-_MAX_RECENT:]


def _pick_varied_index(
    logits: torch.Tensor, candidates: list[dict[str, Any]], username: str
) -> int:
    """Sample from the model ranking while excluding recently served positions.

    Masking the softmax is important: simply zeroing the top probability would
    restart from the same argmax when the model is highly confident, which is
    exactly the \"Generate New Scenario never changes\" bug.
    """
    fens = [c["fen"] for c in candidates]
    scores = torch.nan_to_num(logits[0].float(), nan=0.0, posinf=1e6, neginf=-1e6)
    recent = set(_RECENT_SCENARIOS.get(username, []))
    unseen = [i for i, fen in enumerate(fens) if fen not in recent]
    if not unseen:
        # Every candidate has been seen: start a fresh cycle rather than repeat.
        _RECENT_SCENARIOS[username] = []
        unseen = list(range(len(fens)))

    masked = torch.full_like(scores, -1e9)
    masked[unseen] = scores[unseen]
    probs = torch.softmax(masked, dim=-1)
    if not bool(torch.isfinite(probs).all()) or probs.sum().item() <= 0:
        probs = torch.zeros_like(scores)
        probs[unseen] = 1.0 / len(unseen)

    idx = int(torch.multinomial(probs, 1).item())
    _remember_fen(username, fens[idx])
    return idx


def _pick_varied_candidate(
    candidates: list[dict[str, Any]],
    username: str,
    preferred: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Non-neural equivalent of _pick_varied_index (baseline / fallback)."""
    recent = set(_RECENT_SCENARIOS.get(username, []))
    pool = [c for c in candidates if c["fen"] not in recent] or list(candidates)
    if preferred is not None:
        chosen = next(
            (c for c in pool if c["fen"] == preferred.get("fen")), pool[0]
        )
    else:
        chosen = random.choice(pool)
    _remember_fen(username, chosen["fen"])
    return chosen


@router.post("/predict", response_model=ScenarioResponse)
def predict_scenario(req: ScenarioPredictRequest):
    """
    Predicts a scenario conditioned on the player's history. The returned
    `target_elo` is part of the scenario (derived from its opponent band and
    clock pressure), and consecutive calls avoid recently served positions.
    """
    username = (req.username or "").strip().lower()
    if not username:
        raise HTTPException(
            status_code=422,
            detail="Enter your Chess.com username before generating a scenario.",
        )

    manager = CandidateBankManager.for_user(username)
    candidates = _load_bank(username, manager)

    profile = _profile_for(username)
    player_rating = int(profile.get("player_rating", req.player_rating)) if profile else int(req.player_rating)
    prefix = "[Personalized]"

    if req.use_baseline:
        baseline.fit(candidates)
        matched = baseline.predict(player_rating=player_rating, avg_clock=60.0)
        selected = _pick_varied_candidate(candidates, username, preferred=matched)
        fen, color = _align_color(selected["fen"], selected.get("player_color", "white"))
        clock_bucket = selected.get("clock_bucket", "60-120s")
        rating_band = selected.get("opponent_rating_band", "1200-1400")
        return ScenarioResponse(
            scenario_id=str(uuid.uuid4())[:8],
            fen=fen,
            player_color=color,
            clock_remaining=float(selected.get("clock_remaining", 60.0)),
            clock_bucket=clock_bucket,
            opponent_rating_band=rating_band,
            target_elo=_scenario_target_elo(selected, player_rating, clock_bucket, rating_band),
            reason=f"{prefix} [Baseline KNN] {selected.get('reason', 'Scenario matched from rating profile')}",
        )

    # Neural retrieval. Prefer the player's private fine-tuned model.
    personal_path = USER_MODEL_DIR / f"{username}.pt"
    model_path = personal_path if personal_path.exists() else MODELS_DIR / "scenario_model.pt"

    if model_path.exists():
        try:
            model = ScenarioNetwork(
                board_embedding_dim=settings.EMBEDDING_DIM,
                history_hidden_dim=settings.HISTORY_HIDDEN_DIM,
            )
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            version = _bank_version(candidates)
            cache_key = f"{model_path}:{version}"
            cand_embeddings = _EMBEDDING_CACHE.get(cache_key)
            if cand_embeddings is None:
                cand_embeddings = manager.compute_embeddings(model.board_encoder, device="cpu")
                _EMBEDDING_CACHE[cache_key] = cand_embeddings
            else:
                manager.embeddings = cand_embeddings

            history = _history_tensor(req, model, username)
            with torch.no_grad():
                outputs = model(history, cand_embeddings)
                best_idx = _pick_varied_index(
                    outputs["retrieval_logits"], candidates, username
                )

            best_idx = min(best_idx, len(candidates) - 1)
            selected = candidates[best_idx]
            fen, color = _align_color(selected["fen"], selected.get("player_color", "white"))
            # Use the selected game's real clock/opponent metadata so the returned
            # scenario (and its Stockfish Elo) describes this exact position.
            clock_bucket = selected.get("clock_bucket", "60-120s")
            rating_band = selected.get("opponent_rating_band", "1200-1400")

            return ScenarioResponse(
                scenario_id=str(uuid.uuid4())[:8],
                fen=fen,
                player_color=color,
                clock_remaining=float(selected.get("clock_remaining", 60.0)),
                clock_bucket=clock_bucket,
                opponent_rating_band=rating_band,
                target_elo=_scenario_target_elo(selected, player_rating, clock_bucket, rating_band),
                reason=(
                    f"{prefix} [Neural Model"
                    + (" - private" if personal_path.exists() else "")
                    + f"] {selected.get('reason', 'High evaluation deterioration stretch')}"
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive serving fallback
            import logging

            logging.getLogger(__name__).warning("Neural prediction failed: %s", exc)

    # Fallback: still only this player's own bank.
    selected = _pick_varied_candidate(candidates, username, preferred=candidates[0])
    fen, color = _align_color(selected["fen"], selected.get("player_color", "white"))
    clock_bucket = selected.get("clock_bucket", "60-120s")
    rating_band = selected.get("opponent_rating_band", "1200-1400")
    return ScenarioResponse(
        scenario_id=str(uuid.uuid4())[:8],
        fen=fen,
        player_color=color,
        clock_remaining=float(selected.get("clock_remaining", 60.0)),
        clock_bucket=clock_bucket,
        opponent_rating_band=rating_band,
        target_elo=_scenario_target_elo(selected, player_rating, clock_bucket, rating_band),
        reason=f"{prefix} {selected.get('reason', 'Critical game continuation')}",
    )
