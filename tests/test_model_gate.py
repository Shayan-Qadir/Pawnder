"""
Model Quality Gate Tests
Strict assertions for automated CI/CD gating:
1. Scenario legality must be 100%.
2. Average inference latency must be < 150ms.
3. Clock bucket prediction must be valid.
"""
import time
import chess
import torch
import pytest
from src.models.scenario_network import ScenarioNetwork
from src.training.dataset import CandidateBankManager, fen_to_tensor
from src.config import settings

def test_model_gate_legality_and_latency():
    bank_mgr = CandidateBankManager()
    candidates = bank_mgr.load_candidates()
    if not candidates:
        candidates = [
            {
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                "player_color": "white",
                "clock_remaining": 65.0,
                "clock_bucket": "60-120s",
                "opponent_rating_band": "1200-1400",
            }
        ]
        bank_mgr.save_candidates(candidates)

    model = ScenarioNetwork(
        board_embedding_dim=settings.EMBEDDING_DIM,
        history_hidden_dim=settings.HISTORY_HIDDEN_DIM,
    )
    model.eval()

    cand_embeddings = bank_mgr.compute_embeddings(model.board_encoder, device="cpu")

    # Run 10 inference iterations to measure latency
    latencies = []
    for _ in range(10):
        dummy_input = torch.randn((1, 5, 131))
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(dummy_input, cand_embeddings)
            best_idx = int(torch.argmax(outputs["retrieval_logits"], dim=-1)[0].item())
        latencies.append((time.perf_counter() - start) * 1000)

        # 1. Assert Legality Gate (100% legal FEN)
        retrieved_candidate = candidates[best_idx]
        board = chess.Board(retrieved_candidate["fen"])
        assert board.is_valid(), f"Retrieved FEN {retrieved_candidate['fen']} is illegal!"

    avg_latency = sum(latencies) / len(latencies)
    # 2. Assert Latency Gate (< 150 ms)
    assert avg_latency < 150.0, f"Average inference latency {avg_latency:.2f}ms exceeds 150ms gate limit!"

    # 3. Assert Output Head Bounds
    assert outputs["clock_logits"].shape[-1] == len(settings.CLOCK_BUCKETS)
    assert outputs["rating_logits"].shape[-1] == len(settings.RATING_BANDS)
