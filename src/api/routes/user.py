"""
User Routes: Player Import, Analysis, and Weakness Profiling

Chess.com is the main player-facing platform. Every named player gets a private
scenario bank, profile, and a private model fine-tuned from the Lichess-pretrained
base checkpoint. Nothing imported by one player is served to another.

Lichess is deliberately NOT a user import source here: it is the base dataset
used by the DVC pretraining stages (see src/data/lichess_base_*).

Re-importing an existing player merges newly available Chess.com games into the
cache and analyzes ONLY games that arrived since the previous import. When that
produces new data, a brand-new private model is fine-tuned from the Lichess base
checkpoint; when nothing changed, the player's cached private model is reused so
the import finishes in ~1-2 s. Games already seen are never re-scanned and the
private bank does not inflate.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter

from src.api.schemas import UserImportRequest, UserProfileResponse
from src.data.chess_com_client import ChessComClient
from src.data.game_parser import GameParser
from src.data.blunder_detector import BlunderDetector
from src.engine.stockfish_manager import StockfishManager
from src.training.dataset import CandidateBankManager
from src.config import USER_PROFILE_DIR, RAW_DATA_DIR, USER_MODEL_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/user", tags=["User"])

client = ChessComClient()

# Number of not-yet-scanned games to analyze with Stockfish per import.
MAX_IMPORT_ANALYSIS_GAMES = 12
# Keep a bounded record of which games have already been scanned.
MAX_REMEMBERED_GAMES = 1000
# A repeated import within this window reuses the last live Chess.com fetch, so
# clicking Import twice in a row does not pay for another slow archive walk.
LIVE_FETCH_TTL_SECONDS = 120
LIVE_FETCH_CACHE: dict[tuple[str, int], tuple[float, list[dict[str, Any]]]] = {}

# Lazily initialized so importing this module (e.g. during test collection)
# does not spawn an engine process.
_analysis_engine: Optional[StockfishManager] = None


def _get_analysis_engine() -> Optional[StockfishManager]:
    global _analysis_engine
    if _analysis_engine is None:
        try:
            _analysis_engine = StockfishManager(default_depth=6)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Stockfish unavailable for import analysis: %s", exc)
    return _analysis_engine


def _profile_path(username: str) -> Path:
    return USER_PROFILE_DIR / f"{username}.json"


def _load_profile_from_disk(username: str) -> Optional[dict[str, Any]]:
    path = _profile_path(username)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not load profile for %s: %s", username, exc)
        return None


def _save_profile(username: str, profile: dict[str, Any]) -> None:
    try:
        with open(_profile_path(username), "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not persist profile for %s: %s", username, exc)


def _private_blunders(username: str) -> list[dict[str, Any]]:
    """This player's previously extracted blunders (never another player's)."""
    bank_mgr = CandidateBankManager.for_user(username)
    return [
        dict(candidate)
        for candidate in bank_mgr.load_candidates()
        if candidate.get("owner", username) == username
    ]


def _game_key(game: dict[str, Any]) -> str:
    """Stable identity for a Chess.com game, used to track newly added games."""
    return str(
        game.get("url")
        or game.get("uuid")
        or game.get("pgn", "")[:200]
    )


def _is_loss(raw: dict[str, Any], username: str) -> bool:
    result = raw.get("result")
    white = str((raw.get("white") or {}).get("username", "")).lower()
    black = str((raw.get("black") or {}).get("username", "")).lower()
    return (result == "0-1" and white == username) or (
        result == "1-0" and black == username
    )


def _merge_games(*groups: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge live + cached games, newest source first, deduplicated by game id."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for game in group or []:
            key = _game_key(game)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(game)
    return merged


def _player_rating(parsed_games: list[dict[str, Any]], username: str) -> float:
    """Most recent known rating for this player, with a sensible default."""
    for game in reversed(parsed_games):
        color = game.get("player_color")
        rating = game.get("white_rating") if color == "white" else game.get("black_rating")
        if rating:
            return float(rating)
    return 1300.0


def _fetch_games_cached(username: str, max_months: int) -> list[dict[str, Any]]:
    """Short-lived cache for the slow Chess.com archive walk."""
    key = (username, max_months)
    hit = LIVE_FETCH_CACHE.get(key)
    now = time.monotonic()
    if hit and now - hit[0] < LIVE_FETCH_TTL_SECONDS:
        return list(hit[1])
    try:
        games = client.fetch_player_games(username, max_months=max_months) or []
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Chess.com fetch failed for %s: %s", username, e)
        games = []
    LIVE_FETCH_CACHE[key] = (now, list(games))
    return list(games)


# In-memory session store for imported users (profile summary is also on disk).
USER_CACHE: dict[str, dict[str, Any]] = {}


@router.post("/import", response_model=UserProfileResponse)
def import_user(req: UserImportRequest):
    """
    Import this player's latest Chess.com games and scan only games that are new
    since the previous import for blunders. When new games (or a new bank) exist,
    train a fresh private model from the Lichess base checkpoint. Otherwise reuse
    this player's cached private model so a no-op re-import is fast.
    """
    username = req.username.strip().lower()
    source = "chesscom"  # Chess.com is the only player-facing platform.
    prev_profile = USER_CACHE.get(username) or _load_profile_from_disk(username)

    # Read the old cache BEFORE fetching: fetch_player_games overwrites it with
    # only the current window, and we want to keep older games we already have.
    cached_games = client.load_cached_games(username) or []
    live_games = _fetch_games_cached(username, req.max_months)
    previous_keys = {_game_key(g) for g in cached_games}
    newly_available = [g for g in live_games if _game_key(g) not in previous_keys]

    # Fast path: if nothing changed and all this player's artifacts exist,
    # reuse the cached profile/model. This turns a repeated import from ~11 s
    # (parse + pair build + train) into a few milliseconds.
    if (
        prev_profile
        and not newly_available
        and _private_blunders(username)
        and (USER_MODEL_DIR / f"{username}.pt").exists()
        and prev_profile.get("personalized_model")
    ):
        fast_profile = dict(prev_profile)
        fast_profile["model_reused"] = True
        USER_CACHE[username] = fast_profile
        return fast_profile

    games = _merge_games(live_games, cached_games)
    if games:
        try:
            with open(RAW_DATA_DIR / f"{username}_games.json", "w", encoding="utf-8") as f:
                json.dump(games, f, indent=2)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not refresh game cache for %s: %s", username, exc)

    parsed_games: list[dict[str, Any]] = []
    parsed_by_key: dict[str, dict[str, Any]] = {}
    for raw in games:
        pgn = raw.get("pgn", "")
        if not pgn:
            continue
        parsed = GameParser.parse_pgn(pgn, target_username=username)
        if parsed:
            parsed_games.append(parsed)
            parsed_by_key[_game_key(raw)] = parsed

    existing_blunders = _private_blunders(username)
    current_keys = {_game_key(g) for g in games}
    analyzed: set[str] = {
        key
        for key in (prev_profile or {}).get("analyzed_games", [])
        if key in current_keys
    }

    # Only games that genuinely arrived since the previous import are eligible.
    # Using the previous cache snapshot (rather than "every game not yet in the
    # analyzed list") is important: a partially recorded analyzed list used to
    # make every re-import re-scan up to 12 old games, inflating the bank and
    # adding ~40 s to each import.
    if existing_blunders:
        scan_pool = newly_available
    else:
        # First import (or no private bank yet): bootstrap from recent games.
        scan_pool = games

    ranked = [g for g in scan_pool if _is_loss(g, username)] + [
        g for g in scan_pool if not _is_loss(g, username)
    ]
    targets = ranked[:MAX_IMPORT_ANALYSIS_GAMES]

    fresh_blunders: list[dict[str, Any]] = []
    if targets:
        detector = BlunderDetector(engine_evaluator=_get_analysis_engine())
        for raw in targets:
            key = _game_key(raw)
            # Always record the scan, even when no blunder is extracted, so the
            # same game is not analyzed again on the next import.
            analyzed.add(key)
            parsed = parsed_by_key.get(key)
            if not parsed:
                continue
            fresh_blunders.extend(detector.extract_scenarios_from_game(parsed))
    analyzed &= current_keys
    analyzed = set(sorted(analyzed)[-MAX_REMEMBERED_GAMES:])

    # Merge brand-new scenarios into the private bank, deduplicated by FEN.
    all_blunders = list(existing_blunders)
    seen_fens = {b["fen"] for b in all_blunders}
    for blunder in fresh_blunders:
        if blunder["fen"] not in seen_fens:
            all_blunders.append(blunder)
            seen_fens.add(blunder["fen"])

    if not parsed_games:
        # Offline / brand-new account: a single generic example so the drill
        # flow still works, explicitly owned by this user.
        sample_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        if sample_fen not in seen_fens:
            all_blunders.append(
                {
                    "fen": sample_fen,
                    "player_color": "white",
                    "clock_remaining": 38.0,
                    "clock_bucket": "30-60s",
                    "opponent_rating": 1380,
                    "opponent_rating_band": "1200-1400",
                    "blunder_move": "Nxe4",
                    "cp_drop": 210.0,
                    "win_prob_drop": 0.28,
                    "reason": "Time scramble blunder (<45s)",
                    "ply": 42,
                }
            )
        total_games_count = 0
        losses_count = 0
    else:
        total_games_count = len(parsed_games)
        losses_count = sum(1 for g in parsed_games if g.get("player_won") == 0)

    player_rating = _player_rating(parsed_games, username)
    for blunder in all_blunders:
        blunder["owner"] = username
        blunder.setdefault("player_rating", player_rating)
        blunder.setdefault("_source", source)

    # Write ONLY to this user's bank.
    bank_mgr = CandidateBankManager.for_user(username)
    bank_mgr.save_candidates(all_blunders)

    # Train from scratch only when this import actually produced new data.
    # A repeated import with no newly available games reuses the existing
    # private checkpoint instead of paying ~15 s for an identical retrain.
    model_path = USER_MODEL_DIR / f"{username}.pt"
    personalized = model_path.exists() and bool(existing_blunders)
    model_reused = personalized
    needs_training = (
        not existing_blunders
        or not model_path.exists()
        or bool(newly_available)
        or bool(fresh_blunders)
    )
    training_pairs = int((prev_profile or {}).get("training_pairs", 0))

    if needs_training:
        model_reused = False
        try:
            from src.training.dataset import build_training_pairs_from_bank
            from src.training.train import train_personalized_model

            pairs = build_training_pairs_from_bank(games, username, all_blunders)
            training_pairs = len(pairs)
            result = train_personalized_model(username, all_blunders, pairs)
            personalized = bool(result) or personalized
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Personalized training skipped for %s: %s", username, exc)
    else:
        # Keep the reported pair count honest even on a cached import.
        try:
            from src.training.dataset import build_training_pairs_from_bank

            training_pairs = len(build_training_pairs_from_bank(games, username, all_blunders))
        except Exception:  # pragma: no cover - defensive
            pass

    # Synthesize weakness summary from this player's own scenarios.
    weaknesses = []
    time_trouble = sum(1 for b in all_blunders if "Time scramble" in b.get("reason", ""))
    tactical = sum(1 for b in all_blunders if "Tactical" in b.get("reason", ""))
    endgame = sum(1 for b in all_blunders if "Endgame" in b.get("reason", ""))

    if time_trouble > 0:
        weaknesses.append(f"Severe time-pressure blunder rate ({time_trouble} positions under 45s)")
    if tactical > 0:
        weaknesses.append(f"Calculation oversights under equal material ({tactical} drops >150cp)")
    if endgame > 0:
        weaknesses.append(f"Endgame conversion breakdown after ply 60 ({endgame} positions)")

    if not weaknesses:
        weaknesses = ["Opening tactical stability needs reinforcement under 10+0 Rapid"]

    profile = {
        "username": username,
        "source": source,
        "player_rating": player_rating,
        "total_games": total_games_count,
        "losses_analyzed": losses_count,
        "blunders_extracted": len(all_blunders),
        "games_analyzed": len(analyzed),
        "training_pairs": training_pairs,
        "weaknesses": weaknesses,
        "sample_blunders": all_blunders[:5],
        "personalized_model": personalized,
        "model_reused": model_reused,
        "analyzed_games": sorted(analyzed),
    }

    USER_CACHE[username] = profile
    _save_profile(username, profile)
    return profile


@router.get("/{username}/profile", response_model=UserProfileResponse)
def get_user_profile(username: str):
    uname = username.strip().lower()
    profile = USER_CACHE.get(uname) or _load_profile_from_disk(uname)
    if profile:
        USER_CACHE[uname] = profile
        return profile
    return import_user(UserImportRequest(username=uname, max_months=2))
