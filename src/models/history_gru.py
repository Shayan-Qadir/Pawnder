"""
History GRU Network
Encodes a player's sequence of past games and blunder contexts into a user representation vector.
"""
import torch
import torch.nn as nn


class HistoryGRU(nn.Module):
    """
    Summarizes past games and blunder contexts.
    Input per step: board embedding (128) + context features (clock_ratio, rating_delta, ply) = 131
    """

    def __init__(
        self,
        input_dim: int = 131,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_dim)
        Returns: user_rep: (batch_size, hidden_dim)
        """
        out, _ = self.gru(x)
        # Take the output of the last time step
        last_step = out[:, -1, :]
        user_rep = self.relu(self.fc(last_step))
        return user_rep
