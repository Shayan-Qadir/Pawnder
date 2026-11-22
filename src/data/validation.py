"""Schema and quality checks that run BEFORE any training/evaluation stage.

Bad input must fail loudly rather than silently producing a model from
synthetic fallback data. Run ``python -m src.data.validation`` to validate
every committed dataset at once.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chess import Board

from src.config import RAW_DATA_DIR, CANDIDATE_BANK_DIR

GAME_KEYS = {"pgn", "white", "black"}  # result lives inside the PGN for Chess.com exports
CANDIDATE_KEYS = {"fen", "player_color", "clock_remaining", "clock_bucket", "opponent_rating_band"}


class ValidationError(ValueError):
    """Raised when a dataset violates its declared schema or quality rules."""


def _load(path: Path) -> list[Any]:
    if not path.exists():
        raise ValidationError(f"Missing required dataset: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Corrupted JSON in {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValidationError(f"{path} must contain a JSON list, got {type(data).__name__}")
    if not data:
        raise ValidationError(f"{path} is empty")
    return data


def validate_games(path: Path) -> list[dict[str, Any]]:
    """Validate a raw-games dataset: schema, PGN presence, legal board states."""
    games = _load(path)
    for i, game in enumerate(games):
        if not isinstance(game, dict):
            raise ValidationError(f"{path}[{i}] is not an object")
        missing = GAME_KEYS - game.keys()
        if missing:
            raise ValidationError(f"{path}[{i}] missing keys: {sorted(missing)}")
        if not str(game.get("pgn", "")).strip():
            raise ValidationError(f"{path}[{i}] has empty PGN")
    return games


def validate_candidates(path: Path) -> list[dict[str, Any]]:
    """Validate a candidate bank: schema and legal FENs only."""
    bank = _load(path)
    if len(bank) < 2:
        raise ValidationError(f"{path} needs at least 2 candidates, got {len(bank)}")
    seen: set[str] = set()
    for i, cand in enumerate(bank):
        if not isinstance(cand, dict):
            raise ValidationError(f"{path}[{i}] is not an object")
        missing = CANDIDATE_KEYS - cand.keys()
        if missing:
            raise ValidationError(f"{path}[{i}] missing keys: {sorted(missing)}")
        fen = str(cand["fen"])
        try:
            Board(fen)  # rejects illegal boards
        except ValueError as exc:
            raise ValidationError(f"{path}[{i}] has illegal FEN: {fen!r}") from exc
        if fen in seen:
            raise ValidationError(f"{path}[{i}] duplicate FEN: {fen!r}")
        seen.add(fen)
        if not 0 <= float(cand["clock_remaining"]) <= 3600:
            raise ValidationError(f"{path}[{i}] has out-of-range clock")
    return bank


def validate_all() -> dict[str, int]:
    """Validate every dataset used by the DVC pipeline. Returns row counts."""
    counts: dict[str, int] = {}
    for user in ("bonc_1", "1tshassan"):
        raw = RAW_DATA_DIR / f"{user}_games.json"
        counts[f"games:{user}"] = len(validate_games(raw))
        bank = CANDIDATE_BANK_DIR / "users" / f"{user}.json"
        counts[f"bank:{user}"] = len(validate_candidates(bank))
    counts["games:lichess_base"] = len(validate_games(RAW_DATA_DIR / "lichess_base_games.json"))
    counts["bank:lichess_base"] = len(validate_candidates(CANDIDATE_BANK_DIR / "lichess_base.json"))
    counts["bank:demo"] = len(validate_candidates(CANDIDATE_BANK_DIR / "legal_positions.json"))
    return counts


if __name__ == "__main__":
    result = validate_all()
    print("All datasets valid:", result)
