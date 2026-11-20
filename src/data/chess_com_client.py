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
