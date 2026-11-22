"""
DVC stage: build the Lichess BASE dataset.

Lichess is not a user-facing import platform in Pawnder. Chess.com is. This
stage downloads public Rapid games for a small set of base accounts and tags
each game with the player who lost it, because the blunder detector learns from
lost games. The combined file feeds the base extraction and pretraining stages.

Output: data/raw/lichess_base_games.json
"""
import json
import logging
from typing import Any, Optional

from src.config import RAW_DATA_DIR
from src.data.chess_com_client import load_params
from src.data.lichess_client import LichessClient

logger = logging.getLogger(__name__)

BASE_GAMES_FILE = "lichess_base_games.json"


def losing_username(game: dict[str, Any]) -> Optional[str]:
    """Return the handle of the player who lost a normalized game.

    Draws and unfinished games are skipped: the project learns from losses.
    """
    result = game.get("result")
    if result == "0-1":
        return (game.get("white") or {}).get("username")
    if result == "1-0":
        return (game.get("black") or {}).get("username")
    return None


def ingest_base_games(params: dict, client: Optional[LichessClient] = None) -> list[dict[str, Any]]:
    """Fetch and tag base games for every configured Lichess account."""
    client = client or LichessClient()
    cfg = params.get("base_data", {})
    usernames = [str(u).strip().lower() for u in cfg.get("usernames", []) if str(u).strip()]
    max_games = int(cfg.get("max_games_per_user", 40))
    perf_type = str(cfg.get("perf_type", "rapid"))

    tagged: list[dict[str, Any]] = []
    for username in usernames:
        games = client.load_cached_games(username) or []
        if not games:
            try:
                games = client.fetch_player_games(username, max_games=max_games, perf_type=perf_type)
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("Base fetch failed for Lichess user %s: %s", username, exc)
                games = []
        logger.info("Base dataset: %d Lichess games from %s", len(games), username)

        for game in games:
            loser = losing_username(game)
            if not loser:
                continue
            tagged.append(
                {
                    **game,
                    "_platform": "lichess",
                    "_base_account": username,
                    "_target_username": loser.lower(),
                    "_source": "lichess_base",
                }
            )
    return tagged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    params = load_params()
    games = ingest_base_games(params)

    out_path = RAW_DATA_DIR / BASE_GAMES_FILE
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(games, f, indent=2)
    print(f"Ingested {len(games)} Lichess base games (losses) -> {out_path}")
