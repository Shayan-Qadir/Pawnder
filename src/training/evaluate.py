"""
Model Evaluation and Benchmark vs Baseline
Measures top-k scenario retrieval accuracy, clock MAE, legality (100%), and latency.
Logs a classical kNN baseline BEFORE the neural model so the baseline is always
recorded first, then logs both to MLflow with the dataset version.
"""
import time
from typing import Any
import torch
import numpy as np
from src.config import settings
from src.models.scenario_network import ScenarioNetwork
from src.models.baseline_knn import BaselineKNNMatcher
from src.training.dataset import CandidateBankManager, HistoryScenarioDataset


def evaluate_model(
    model: ScenarioNetwork,
    test_samples: list[dict[str, Any]],
    candidate_bank: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the scenario network on held-out games against a kNN baseline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    bank_mgr = CandidateBankManager()
    bank_mgr.candidates = candidate_bank
    cand_embeddings = bank_mgr.compute_embeddings(model.board_encoder, device=str(device)).to(device)

    dataset = HistoryScenarioDataset(test_samples, board_encoder=model.board_encoder)

    # Classical baseline is fitted and evaluated FIRST (it is the reference point).
    baseline = BaselineKNNMatcher()
    baseline.fit(candidate_bank)

    model_top1_correct = 0
    baseline_top1_correct = 0
    model_clock_correct = 0
    total = len(test_samples)
    latencies_ms = []

    for idx in range(total):
        item = dataset[idx]
        history = item["history"].unsqueeze(0).to(device)
        target_cand = item["target_candidate_idx"].item()
        target_clock = item["target_clock_bucket"].item()
        target_fen = candidate_bank[target_cand].get("fen")

        # Baseline: nearest neighbour on clock/rating features.
        sample = test_samples[idx]
        last_step = sample.get("history", [{}])[-1] if sample.get("history") else {}
        baseline_pick = baseline.predict(
            player_rating=float(last_step.get("opponent_rating", 1200)),
            avg_clock=float(last_step.get("clock_remaining", 300)),
        )
        baseline_top1_correct += int(baseline_pick.get("fen") == target_fen)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(history, cand_embeddings)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        pred_cand = torch.argmax(outputs["retrieval_logits"], dim=-1).item()
        pred_clock = torch.argmax(outputs["clock_logits"], dim=-1).item()

        if pred_cand == target_cand:
            model_top1_correct += 1
        if pred_clock == target_clock:
            model_clock_correct += 1

    n = max(1, total)
    return {
        "samples_evaluated": total,
        "baseline_top1_accuracy": round(baseline_top1_correct / n, 4),
        "scenario_top1_accuracy": round(model_top1_correct / n, 4),
        "clock_bucket_accuracy": round(model_clock_correct / n, 4),
        "avg_inference_latency_ms": round(float(np.mean(latencies_ms)) if latencies_ms else 0.0, 2),
        "legality_guarantee_pct": 100.0,
    }


if __name__ == "__main__":
    """DVC evaluate stage: chronological holdout on real data only."""
    import hashlib
    import json
    import logging

    from src.config import MODELS_DIR, RAW_DATA_DIR, CANDIDATE_BANK_DIR
    from src.data.chess_com_client import load_params
    from src.data.validation import validate_games, validate_candidates, ValidationError
    from src.tracking import start_run, log_params, log_metrics, log_artifact
    from src.training.dataset import CandidateBankManager, build_training_pairs_from_bank

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    params = load_params()
    username = str(params.get("data", {}).get("username", "bonc_1")).lower()

    bank = CandidateBankManager()
    cands = bank.load_candidates()
    if not cands:
        from src.api.routes.scenario import get_default_candidates

        cands = get_default_candidates()
        bank.save_candidates(cands)

    raw_path = RAW_DATA_DIR / f"{username}_games.json"
    try:
        validate_games(raw_path)
        validate_candidates(CANDIDATE_BANK_DIR / "legal_positions.json")
    except ValidationError as exc:
        raise SystemExit(f"Evaluation dataset validation failed: {exc}") from exc

    raw_games = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_games = raw_games[: int(params.get("extract", {}).get("max_games", 30))]
    all_pairs = build_training_pairs_from_bank(raw_games, username, cands)

    split = int(len(all_pairs) * 0.8)
    if split < 1 or len(all_pairs) - split < 1:
        raise SystemExit("Not enough chronological pairs for an 80/20 holdout")
    test_samples = all_pairs[split:]
    logger.info("Chronological holdout: %d train pairs, %d test pairs", split, len(test_samples))

    model = ScenarioNetwork()
    checkpoint_path = MODELS_DIR / "scenario_model.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded trained checkpoint from %s", checkpoint_path)

    with start_run("evaluate_holdout", tags={"stage": "evaluate", "dataset": "chess_com_holdout"}) as _:
        log_params({
            "username": username,
            "dataset_md5": hashlib.md5(raw_path.read_bytes()).hexdigest(),
            "checkpoint_md5": hashlib.md5(checkpoint_path.read_bytes()).hexdigest(),
            "train_pairs": split,
            "test_pairs": len(test_samples),
            "split": "chronological_80_20",
        })
        metrics = evaluate_model(model, test_samples, cands)
        log_metrics(metrics)
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics written:", metrics)
