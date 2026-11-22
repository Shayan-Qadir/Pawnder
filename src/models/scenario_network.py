"""
Joint History-to-Scenario Prediction & Retrieval Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import settings
from src.models.board_encoder import CompactBoardCNN
from src.models.history_gru import HistoryGRU


class ScenarioNetwork(nn.Module):
    """
    Multi-task model that combines player history with candidate bank retrieval,
    clock bucket prediction, and opponent rating band classification.
    """

    def __init__(
        self,
        board_embedding_dim: int = 128,
        history_hidden_dim: int = 256,
        num_clock_buckets: int = 4,
        num_rating_bands: int = 6,
        history_similarity_weight: float = 10.0,
    ):
        super().__init__()
        self.history_similarity_weight = history_similarity_weight
        self.board_encoder = CompactBoardCNN(
            in_channels=settings.BOARD_CHANNELS, embedding_dim=board_embedding_dim
        )
        self.history_gru = HistoryGRU(
            input_dim=board_embedding_dim + 3, hidden_dim=history_hidden_dim
        )

        # Retrieval projection: project user state into board embedding space
        self.retrieval_proj = nn.Linear(history_hidden_dim, board_embedding_dim)

        # Auxiliary Prediction Heads
        self.clock_head = nn.Sequential(
            nn.Linear(history_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_clock_buckets),
        )

        self.rating_head = nn.Sequential(
            nn.Linear(history_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_rating_bands),
        )

    def forward(
        self,
        history_tensors: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            history_tensors: (batch_size, seq_len, 131)
            candidate_embeddings: (num_candidates, board_embedding_dim)
        Returns:
            dict containing:
                retrieval_logits: (batch_size, num_candidates)
                clock_logits: (batch_size, num_clock_buckets)
                rating_logits: (batch_size, num_rating_bands)
                user_embedding: (batch_size, board_embedding_dim)
        """
        user_rep = self.history_gru(history_tensors)  # (batch_size, 256)
        query_vec = self.retrieval_proj(user_rep)  # (batch_size, 128)

        # Cosine similarity for retrieval. The learned query captures the 5-step
        # pattern; the last real blunder's board embedding (history tensors are
        # left-padded, so the final step is always the most recent) is a strong
        # signal for "practice a position that looks like this one". Blending
        # both keeps personalization while avoiding a cold, noisy query.
        norm_query = F.normalize(query_vec, p=2, dim=-1)
        norm_candidates = F.normalize(candidate_embeddings, p=2, dim=-1)
        query_similarity = torch.matmul(norm_query, norm_candidates.T)
        last_board = history_tensors[:, -1, : candidate_embeddings.shape[-1]]
        history_similarity = torch.matmul(
            F.normalize(last_board, p=2, dim=-1), norm_candidates.T
        )
        retrieval_logits = (
            query_similarity + self.history_similarity_weight * history_similarity
        )

        clock_logits = self.clock_head(user_rep)
        rating_logits = self.rating_head(user_rep)

        return {
            "retrieval_logits": retrieval_logits,
            "clock_logits": clock_logits,
            "rating_logits": rating_logits,
            "query_embedding": query_vec,
        }
