"""
Chess.com Public API Client for Pawnder
"""
import json
import logging
from pathlib import Path
from typing import Any, Optional
import requests
from src.config import settings, RAW_DATA_DIR

logger = logging.getLogger(__name__)

PARAMS_PATH = Path(__file__).resolve().parent.parent.parent / "params.yaml"


def load_params() -> dict[str, Any]:
    """Load params.yaml for standalone DVC stage execution."""
    try:
        import yaml

        if PARAMS_PATH.exists():
            with open(PARAMS_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as exc:  # pragma: no cover - defensive for offline runs
        logger.warning("Could not read params.yaml: %s", exc)
    return {}


class ChessComClient:
    """Client to query Chess.com Published Data API."""

    BASE_URL = "https://api.chess.com/pub"

    def __init__(self, user_agent: Optional[str] = None):
        self.headers = {
            "User-Agent": user_agent or settings.CHESS_COM_USER_AGENT,
            "Accept": "application/json",
        }

    def get_player_profile(self, username: str) -> dict[str, Any]:
        """Fetch player profile stats."""
        url = f"{self.BASE_URL}/player/{username.lower()}"
        resp = requests.get(url, headers=self.headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_monthly_archives(self, username: str) -> list[str]:
        """Fetch list of monthly archive URLs."""
        url = f"{self.BASE_URL}/player/{username.lower()}/games/archives"
        resp = requests.get(url, headers=self.headers, timeout=10)
        resp.raise_for_status()
        return resp.json().get("archives", [])

    def get_games_for_month(self, archive_url: str) -> list[dict[str, Any]]:
        """Fetch games for a specific month URL."""
        resp = requests.get(archive_url, headers=self.headers, timeout=15)
        resp.raise_for_status()
        return resp.json().get("games", [])

    def fetch_player_games(
        self, username: str, max_months: int = 3, time_class: str = "rapid"
    ) -> list[dict[str, Any]]:
        """
        Fetch completed games for the player, filtered by time_class (e.g., 'rapid').
        Saves downloaded raw data to raw cache.
        """
        archives = self.get_monthly_archives(username)
        recent_archives = archives[-max_months:] if archives else []

        all_games = []
        for archive_url in recent_archives:
            try:
                games = self.get_games_for_month(archive_url)
                for g in games:
                    if g.get("time_class") == time_class:
                        all_games.append(g)
            except Exception as e:
                logger.warning(f"Failed to fetch games from {archive_url}: {e}")

        # Cache locally
        user_cache_file = RAW_DATA_DIR / f"{username.lower()}_games.json"
        with open(user_cache_file, "w", encoding="utf-8") as f:
            json.dump(all_games, f, indent=2)

        return all_games

    def load_cached_games(self, username: str) -> Optional[list[dict[str, Any]]]:
        """Load locally cached games if available."""
        user_cache_file = RAW_DATA_DIR / f"{username.lower()}_games.json"
        if user_cache_file.exists():
            with open(user_cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None


def _offline_sample_games(username: str) -> list[dict[str, Any]]:
    """
    Clearly-labeled, deterministic sample games used only when the Chess.com
    API is unreachable. Keeps the DVC pipeline runnable offline.
    """
    pgn = """[Event "Live Chess"]
[Site "Chess.com"]
[Date "2026.10.01"]
[White "{user}"]
[Black "practice_opponent"]
[Result "0-1"]
[WhiteElo "1350"]
[BlackElo "1420"]
[TimeControl "600"]

1. e4 {{[%clk 0:09:58]}} 1... e5 {{[%clk 0:09:55]}} 2. Nf3 {{[%clk 0:09:40]}} 2... Nc6 {{[%clk 0:09:30]}}
3. Bc4 {{[%clk 0:09:12]}} 3... Bc5 {{[%clk 0:09:00]}} 4. Nxe5 {{[%clk 0:00:41]}} 4... Nxe5 {{[%clk 0:00:38]}} 0-1
""".format(user=username)
    return [
        {
            "url": "https://www.chess.com/game/live/offline-sample",
            "time_class": "rapid",
            "time_control": "600",
            "white": {"username": username, "rating": 1350},
            "black": {"username": "practice_opponent", "rating": 1420},
            "result": "0-1",
            "pgn": pgn,
            "_source": "offline_sample",
        }
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    params = load_params()
    data_params = params.get("data", {})
    username = str(data_params.get("username", "bonc_1"))
    max_months = int(data_params.get("max_months", 3))

    client = ChessComClient()
    games = client.fetch_player_games(username, max_months=max_months)
    if not games:
        cached = client.load_cached_games(username)
        if cached:
            games = cached
            logger.info("Using %d cached games for %s", len(games), username)
    if not games:
        games = _offline_sample_games(username)
        logger.warning("Live fetch returned no games; writing labeled offline sample")

    out_file = RAW_DATA_DIR / f"{username.lower()}_games.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(games, f, indent=2)
    print(f"Ingested {len(games)} games -> {out_file}")
