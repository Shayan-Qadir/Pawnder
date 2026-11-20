"""
PyTorch Dataset and Candidate Bank Manager for History-Scenario Modeling
"""
import json
from pathlib import Path
from typing import Any, Optional
import torch
from torch.utils.data import Dataset
from src.config import settings, CANDIDATE_BANK_DIR
from src.models.board_encoder import fen_to_tensor, CompactBoardCNN


class CandidateBankManager:
    """Manages legal position candidate bank and their pre-computed embeddings."""

    def __init__(self, bank_path: Optional[Path] = None):
        self.bank_path = bank_path or (CANDIDATE_BANK_DIR / "legal_positions.json")
        self.embeddings_path = CANDIDATE_BANK_DIR / "candidate_embeddings.pt"
        self.candidates: list[dict[str, Any]] = []
        self.embeddings: Optional[torch.Tensor] = None

    def save_candidates(self, candidates: list[dict[str, Any]]) -> None:
        self.candidates = candidates
        with open(self.bank_path, "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2)

    def load_candidates(self) -> list[dict[str, Any]]:
        if self.bank_path.exists():
            with open(self.bank_path, "r", encoding="utf-8") as f:
                self.candidates = json.load(f)
        return self.candidates

    def compute_embeddings(
        self, board_encoder: CompactBoardCNN, device: str = "cpu"
    ) -> torch.Tensor:
        """Precomputes CNN board embeddings for all candidates in bank."""
        if not self.candidates:
            self.load_candidates()

        if not self.candidates:
            # Fallback seed candidate
            self.candidates = [
                {
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "player_color": "white",
                    "clock_remaining": 600.0,
                    "clock_bucket": "120s+",
                    "opponent_rating_band": "1200-1400",
                }
            ]

        board_encoder.eval()
        board_encoder.to(device)
        tensors = []
        for c in self.candidates:
            t = fen_to_tensor(c["fen"])
            tensors.append(t)

        batch_tensors = torch.stack(tensors).to(device)
        with torch.no_grad():
            self.embeddings = board_encoder(batch_tensors).cpu()

        torch.save(self.embeddings, self.embeddings_path)
        return self.embeddings

    def get_embeddings(self) -> torch.Tensor:
        if self.embeddings is None:
            if self.embeddings_path.exists():
                self.embeddings = torch.load(self.embeddings_path, map_location="cpu", weights_only=True)
            else:
                encoder = CompactBoardCNN()
                self.embeddings = self.compute_embeddings(encoder)
        return self.embeddings


class HistoryScenarioDataset(Dataset):
    """
    Dataset pairing earlier game blunder history with later target practice scenario.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        board_encoder: Optional[CompactBoardCNN] = None,
        max_seq_len: int = 5,
    ):
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.board_encoder = board_encoder or CompactBoardCNN()
        self.board_encoder.eval()

        self.clock_map = {b: i for i, b in enumerate(settings.CLOCK_BUCKETS)}
        self.rating_map = {r: i for i, r in enumerate(settings.RATING_BANDS)}

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_step(self, step_info: dict[str, Any]) -> torch.Tensor:
        """Encodes single history step: (128-d board + 3 context features)."""
        fen = step_info.get("fen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        board_t = fen_to_tensor(fen).unsqueeze(0)
        with torch.no_grad():
            board_emb = self.board_encoder(board_t).squeeze(0)  # (128,)

        clock = float(step_info.get("clock_remaining", 300.0))
        clock_ratio = min(1.0, max(0.0, clock / 600.0))

        p_rating = float(step_info.get("player_rating", 1200.0))
        o_rating = float(step_info.get("opponent_rating", 1200.0))
        rating_delta = (p_rating - o_rating) / 400.0

        ply = float(step_info.get("ply", 30.0)) / 100.0

        ctx = torch.tensor([clock_ratio, rating_delta, ply], dtype=torch.float32)
        return torch.cat([board_emb, ctx])  # (131,)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]
        history = item.get("history", [])

        # Construct sequence
        step_tensors = []
        for h in history[-self.max_seq_len :]:
            step_tensors.append(self._encode_step(h))

        # Pad if needed
        while len(step_tensors) < self.max_seq_len:
            step_tensors.insert(0, torch.zeros(131, dtype=torch.float32))

        history_tensor = torch.stack(step_tensors)  # (max_seq_len, 131)

        candidate_idx = item.get("target_candidate_idx", 0)
        clock_bucket = item.get("target_clock_bucket", "60-120s")
        rating_band = item.get("target_rating_band", "1200-1400")

        return {
            "history": history_tensor,
            "target_candidate_idx": torch.tensor(candidate_idx, dtype=torch.long),
            "target_clock_bucket": torch.tensor(
                self.clock_map.get(clock_bucket, 0), dtype=torch.long
            ),
            "target_rating_band": torch.tensor(
                self.rating_map.get(rating_band, 0), dtype=torch.long
            ),
        }
