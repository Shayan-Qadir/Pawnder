"""
DVC ingest stage for the MAIN platform: Chess.com.

Lichess game data is handled separately by lichess_base_ingest.py, which builds
the base dataset used to pretrain the model. Player-facing imports always use
Chess.com.

Output: data/raw/{username}_games.json
"""
import json
import logging

from src.config import RAW_DATA_DIR
from src.data.chess_com_client import ChessComClient, load_params, _offline_sample_games

logger = logging.getLogger(__name__)


def ingest_games(username: str, max_months: int = 3) -> list[dict]:
    username = username.strip().lower()
    client = ChessComClient()
    games = client.fetch_player_games(username, max_months=max_months)
    if not games:
        games = client.load_cached_games(username) or []
    if not games:
        games = _offline_sample_games(username)
        logger.warning("Live fetch returned no games; writing labeled offline sample")

    out_file = RAW_DATA_DIR / f"{username}_games.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(games, f, indent=2)
    return games


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_params = load_params().get("data", {})
    games = ingest_games(
        str(data_params.get("username", "bonc_1")),
        max_months=int(data_params.get("max_months", 3)),
    )
    print(f"Ingested {len(games)} Chess.com games -> {RAW_DATA_DIR}")
