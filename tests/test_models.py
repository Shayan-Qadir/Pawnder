import torch
import pytest
from src.models.board_encoder import fen_to_tensor, CompactBoardCNN
from src.models.history_gru import HistoryGRU
from src.models.scenario_network import ScenarioNetwork

def test_fen_to_tensor():
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tensor = fen_to_tensor(start_fen)
    assert tensor.shape == (14, 8, 8)
    assert tensor.dtype == torch.float32

def test_board_encoder_forward():
    encoder = CompactBoardCNN(in_channels=14, embedding_dim=128)
    dummy_input = torch.zeros((2, 14, 8, 8))
    out = encoder(dummy_input)
    assert out.shape == (2, 128)

def test_history_gru_forward():
    gru = HistoryGRU(input_dim=131, hidden_dim=256)
    dummy_seq = torch.zeros((2, 5, 131))
    out = gru(dummy_seq)
    assert out.shape == (2, 256)

def test_scenario_network_forward():
    model = ScenarioNetwork(board_embedding_dim=128, history_hidden_dim=256)
    history_batch = torch.zeros((2, 5, 131))
    candidates = torch.randn((10, 128))
    outputs = model(history_batch, candidates)
    
    assert outputs["retrieval_logits"].shape == (2, 10)
    assert outputs["clock_logits"].shape == (2, 4)
    assert outputs["rating_logits"].shape == (2, 6)
