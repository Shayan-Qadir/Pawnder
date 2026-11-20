"""
Baseline Nearest-Neighbor Scenario Matcher
Serves as the benchmark baseline against the neural architecture.
"""
from typing import Any
import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.config import settings


class BaselineKNNMatcher:
    """Matches user profile to scenario bank using handcrafted features."""

    def __init__(self, n_neighbors: int = 1):
        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self.candidate_scenarios: list[dict[str, Any]] = []
        self.is_fitted = False

    def _extract_features(
        self, player_rating: float, avg_clock: float, loss_rate: float
    ) -> np.ndarray:
        """Handcrafted feature vector: [scaled_rating, scaled_clock, loss_rate]."""
        return np.array(
            [player_rating / 2000.0, avg_clock / 600.0, loss_rate], dtype=np.float32
        )

    def fit(self, candidate_bank: list[dict[str, Any]]) -> None:
        """Fit kNN index on candidate scenarios."""
        self.candidate_scenarios = candidate_bank
        if not candidate_bank:
            return

        feature_matrix = []
        for s in candidate_bank:
            opp_rating = float(s.get("opponent_rating") or 1200)
            clock = float(s.get("clock_remaining") or 300)
            feature_matrix.append(self._extract_features(opp_rating, clock, 0.5))

        feature_matrix = np.array(feature_matrix)
        self.knn.fit(feature_matrix)
        self.is_fitted = True

    def predict(
        self, player_rating: float, avg_clock: float, loss_rate: float = 0.5
    ) -> dict[str, Any]:
        """Predict nearest scenario."""
        if not self.is_fitted or not self.candidate_scenarios:
            # Fallback default scenario
            return {
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                "player_color": "white",
                "clock_remaining": 60.0,
                "clock_bucket": "60-120s",
                "opponent_rating_band": "1200-1400",
                "reason": "Baseline fallback scenario",
            }

        q_vec = self._extract_features(player_rating, avg_clock, loss_rate).reshape(1, -1)
        distances, indices = self.knn.kneighbors(q_vec)
        best_idx = indices[0][0]
        match = self.candidate_scenarios[best_idx].copy()
        match["baseline_distance"] = float(distances[0][0])
        return match
