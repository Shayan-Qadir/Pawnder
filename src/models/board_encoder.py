"""
Compact CNN Board Encoder (8x8x14)
Encodes a chess board position into a low-dimensional embedding vector.
"""
import torch
import torch.nn as nn
import chess


def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Converts a FEN string into an 8x8x14 binary PyTorch tensor.
    Planes 0-5: White P, N, B, R, Q, K
    Planes 6-11: Black p, n, b, r, q, k
    Plane 12: Turn (1 if White, 0 if Black)
    Plane 13: Castling rights (White kingside, queenside, Black kingside, queenside)
    """
    board = chess.Board(fen)
    tensor = torch.zeros((14, 8, 8), dtype=torch.float32)

    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        base_idx = 0 if piece.color == chess.WHITE else 6
        plane_idx = base_idx + piece_map[piece.piece_type]
        tensor[plane_idx, row, col] = 1.0

    # Plane 12: Turn
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # Plane 13: Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, 0, 0:2] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, 0, 2:4] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[13, 7, 0:2] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[13, 7, 2:4] = 1.0

    return tensor


class CompactBoardCNN(nn.Module):
    """Compact CNN to encode board tensor into embedding vector."""

    def __init__(self, in_channels: int = 14, embedding_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 14, 8, 8)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
