"""
Lichess Public API Client for Pawnder.

Uses the Lichess game export endpoint, which streams NDJSON. We request the
full PGN plus clock comments with ``pgnInJson=true`` and ``clocks=true``; the
resulting PGNs use the same ``[%clk ...]`` annotations as Chess.com, so the
existing GameParser works unchanged.
"""
import io
import json
import logging
from pathlib import Path
from typing import Any, Optional

import chess.pgn
import requests

from src.config import settings, RAW_DATA_DIR

logger = logging.getLogger(__name__)

PARAMS_PATH = Path(__file__).resolve().parent.parent.parent / "params.yaml"


class LichessClient:
    """Client for the public Lichess API (no token required for public data)."""

    BASE_URL = "https://lichess.org/api"

    def __init__(self, user_agent: Optional[str] = None, timeout: int = 60):
        self.headers = {
            "User-Agent": user_agent or settings.CHESS_COM_USER_AGENT,
            "Accept": "application/x-ndjson",
        }
        self.timeout = timeout

    def get_player_profile(self, username: str) -> dict[str, Any]:
        resp = requests.get(
            f"{self.BASE_URL}/user/{username.lower()}",
            headers={"User-Agent": self.headers["User-Agent"], "Accept": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def fetch_player_games(
        self,
        username: str,
        max_games: int = 40,
        perf_type: str = "rapid",
    ) -> list[dict[str, Any]]:
        """Download recent finished games and normalize them to our raw format."""
        params = {
            "max": max(1, int(max_games)),
            "perfType": perf_type,
            "clocks": "true",
            "pgnInJson": "true",
            "moves": "true",
            "tags": "true",
            "finished": "true",
        }
        url = f"{self.BASE_URL}/games/user/{username.lower()}"
        logger.info("Fetching up to %d %s games from Lichess for %s", max_games, perf_type, username)

        games: list[dict[str, Any]] = []
        with requests.get(
            url, headers=self.headers, params=params, stream=True, timeout=(10, self.timeout)
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:  # pragma: no cover - defensive
                    logger.warning("Skipping malformed Lichess NDJSON line")
                    continue
                normalized = self._normalize_game(payload)
                if normalized is not None:
                    games.append(normalized)

        self._cache_games(username, games)
        return games

    @staticmethod
    def _normalize_game(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        pgn = payload.get("pgn")
        if not pgn:
            return None

        try:
            game = chess.pgn.read_game(io.StringIO(pgn))
        except Exception:  # pragma: no cover - defensive
            return None
        if game is None:
            return None
        headers = game.headers

        players = payload.get("players") or {}
        white = players.get("white") or {}
        black = players.get("black") or {}

        def handle(side: dict[str, Any], header: str) -> str:
            user = side.get("user") or {}
            return str(user.get("name") or headers.get(header, ""))

        return {
            "url": payload.get("url") or f"https://lichess.org/{payload.get('id', '')}",
            "time_class": payload.get("perf") or "rapid",
            "time_control": headers.get("TimeControl", "600+0"),
            "white": {"username": handle(white, "White"), "rating": white.get("rating")},
            "black": {"username": handle(black, "Black"), "rating": black.get("rating")},
            "result": headers.get("Result", "*"),
            "pgn": pgn,
            "_source": "lichess",
        }

    @staticmethod
    def _cache_path(username: str) -> Path:
        return RAW_DATA_DIR / f"lichess_{username.lower()}_games.json"

    def _cache_games(self, username: str, games: list[dict[str, Any]]) -> None:
        with open(self._cache_path(username), "w", encoding="utf-8") as f:
            json.dump(games, f, indent=2)

    def load_cached_games(self, username: str) -> Optional[list[dict[str, Any]]]:
        path = self._cache_path(username)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data.chess_com_client import load_params

    params = load_params()
    data_params = params.get("data", {})
    client = LichessClient()
    games = client.fetch_player_games(
        str(data_params.get("username", "bonc_1")).lower(),
        max_games=int(data_params.get("lichess", {}).get("max_games", 40)),
        perf_type=str(data_params.get("lichess", {}).get("perf_type", "rapid")),
    )
    print(f"Ingested {len(games)} Lichess games")
