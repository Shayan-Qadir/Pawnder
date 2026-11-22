"""
PyTorch Dataset and Candidate Bank Manager for History-Scenario Modeling
"""
import json
from pathlib import Path
from typing import Any, Optional
import torch
from torch.utils.data import Dataset
from src.config import settings, CANDIDATE_BANK_DIR, USER_BANK_DIR
from src.models.board_encoder import fen_to_tensor, CompactBoardCNN


class CandidateBankManager:
    """Manages legal position candidate bank and their pre-computed embeddings."""

    def __init__(
        self, bank_path: Optional[Path] = None, username: Optional[str] = None
    ):
        self.username = username.strip().lower() if username else None
        if bank_path is not None:
            # Explicit banks (e.g. the Lichess base bank) get their own
            # embedding cache so they cannot clobber the default one.
            self.bank_path = Path(bank_path)
            self.embeddings_path = self.bank_path.with_name(
                f"{self.bank_path.stem}_embeddings.pt"
            )
        elif self.username:
            # A named player gets a private bank. Nothing from the global
            # demo bank is mixed into it, so scenarios cannot leak across users.
            self.bank_path = USER_BANK_DIR / f"{self.username}.json"
            self.embeddings_path = USER_BANK_DIR / f"{self.username}_embeddings.pt"
        else:
            self.bank_path = CANDIDATE_BANK_DIR / "legal_positions.json"
            self.embeddings_path = CANDIDATE_BANK_DIR / "candidate_embeddings.pt"
        self.candidates: list[dict[str, Any]] = []
        self.embeddings: Optional[torch.Tensor] = None

    @classmethod
    def for_user(cls, username: str) -> "CandidateBankManager":
        """Return a manager bound to one player's private scenario bank."""
        return cls(username=username)

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
        self,
        board_encoder: CompactBoardCNN,
        device: str = "cpu",
        trainable: bool = False,
    ) -> torch.Tensor:
        """Precomputes CNN board embeddings for all candidates in bank.

        ``trainable=True`` keeps the autograd graph so the encoder can learn
        from the retrieval loss. Inference callers keep the default no-grad path.
        """
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
        if trainable:
            self.embeddings = board_encoder(batch_tensors).cpu()
        else:
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
        trainable: bool = False,
    ):
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.trainable = trainable
        self.board_encoder = board_encoder or CompactBoardCNN()
        self.board_encoder.eval()

        self.clock_map = {b: i for i, b in enumerate(settings.CLOCK_BUCKETS)}
        self.rating_map = {r: i for i, r in enumerate(settings.RATING_BANDS)}

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_step(self, step_info: dict[str, Any]) -> torch.Tensor:
        """Encodes single history step: (128-d board + 3 context features)."""
        return encode_step(step_info, self.board_encoder, trainable=self.trainable)

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


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def encode_step(
    step_info: dict[str, Any], encoder: CompactBoardCNN, trainable: bool = False
) -> torch.Tensor:
    """Encode one history step as 128-d board embedding + 3 context features."""
    fen = step_info.get("fen", START_FEN)
    board_t = fen_to_tensor(fen).unsqueeze(0)
    if trainable:
        board_emb = encoder(board_t).squeeze(0)
    else:
        with torch.no_grad():
            board_emb = encoder(board_t).squeeze(0)

    clock = float(step_info.get("clock_remaining", 300.0))
    clock_ratio = min(1.0, max(0.0, clock / 600.0))
    p_rating = float(step_info.get("player_rating", 1200.0))
    o_rating = float(step_info.get("opponent_rating", 1200.0))
    rating_delta = (p_rating - o_rating) / 400.0
    ply = float(step_info.get("ply", 30.0)) / 100.0

    ctx = torch.tensor([clock_ratio, rating_delta, ply], dtype=torch.float32)
    return torch.cat([board_emb, ctx])


def encode_history_tensor(
    history: list[dict[str, Any]],
    encoder: CompactBoardCNN,
    max_seq_len: int = 5,
) -> torch.Tensor:
    """Build a (1, max_seq_len, 131) inference tensor from a user's blunder history."""
    steps = [encode_step(h, encoder) for h in history[-max_seq_len:]]
    while len(steps) < max_seq_len:
        steps.insert(0, torch.zeros(settings.EMBEDDING_DIM + 3, dtype=torch.float32))
    return torch.stack(steps).unsqueeze(0)


def build_training_pairs(
    raw_games: list[dict[str, Any]],
    username: str,
    bank: list[dict[str, Any]],
    detector: Any,
    player_rating: float = 1300.0,
    max_seq_len: int = 5,
) -> list[dict[str, Any]]:
    """
    Convert parsed game history into (history -> target candidate) pairs.

    Only strictly earlier blunders are used as history for a target blunder,
    which prevents temporal leakage.
    """
    from src.data.game_parser import GameParser

    fen_to_idx = {c["fen"]: i for i, c in enumerate(bank)}
    history: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    for raw in raw_games:
        parsed = GameParser.parse_pgn(raw.get("pgn", ""), target_username=username)
        if parsed is None:
            continue
        scenarios = detector.extract_scenarios_from_game(parsed)
        for scenario in scenarios:
            idx = fen_to_idx.get(scenario["fen"])
            if idx is not None and history:
                pairs.append(
                    {
                        "history": [dict(h, player_rating=player_rating) for h in history[-max_seq_len:]],
                        "target_candidate_idx": idx,
                        "target_clock_bucket": scenario["clock_bucket"],
                        "target_rating_band": scenario["opponent_rating_band"],
                    }
                )
            history.append(dict(scenario, player_rating=player_rating))

    return pairs


def build_training_pairs_from_bank(
    raw_games: list[dict[str, Any]],
    username: Optional[str],
    bank: list[dict[str, Any]],
    max_seq_len: int = 5,
) -> list[dict[str, Any]]:
    """Build chronological pairs from engine-verified scenarios in the candidate bank.

    This is the preferred path for DVC stages: it reuses the output of the
    extraction stage instead of evaluating every move again with Stockfish.

    For the multi-player Lichess base dataset, each raw game can carry its own
    ``_target_username`` (the loser); otherwise ``username`` is used.
    """
    from src.data.game_parser import GameParser

    fen_to_idx = {c["fen"]: i for i, c in enumerate(bank)}
    pairs: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    current_target: Optional[str] = None

    for raw in raw_games:
        target = raw.get("_target_username") or username
        if not target:
            continue
        # Never carry one player's blunder history into another player's games.
        if target != current_target:
            history = []
            current_target = target
        parsed = GameParser.parse_pgn(raw.get("pgn", ""), target_username=target)
        if parsed is None:
            continue
        player_color = parsed.get("player_color")
        for move in parsed.get("moves", []):
            if move.get("turn_color") != player_color:
                continue
            candidate_idx = fen_to_idx.get(move.get("fen_before"))
            if candidate_idx is None:
                continue
            candidate = bank[candidate_idx]
            rich_scenario = dict(candidate)
            rich_scenario.setdefault("clock_remaining", move.get(f"{player_color}_clock", 300.0))
            rich_scenario.setdefault("ply", move.get("ply", 30))
            if history:
                pairs.append(
                    {
                        "history": [dict(h) for h in history[-max_seq_len:]],
                        "target_candidate_idx": candidate_idx,
                        "target_clock_bucket": candidate.get(
                            "clock_bucket", settings.CLOCK_BUCKETS[0]
                        ),
                        "target_rating_band": candidate.get(
                            "opponent_rating_band", settings.RATING_BANDS[0]
                        ),
                    }
                )
            history.append(rich_scenario)

    return pairs

