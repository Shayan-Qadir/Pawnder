"""
Offline usefulness evaluation for the personalized scenario ranker.

Question answered here: on games that come *after* the training games, does the
private model choose a drill of the same failure type and similar board shape
more often than random, most-recent, majority-class or a hand-written rule?

The evaluated model is fine-tuned only on the earlier 80% of each player's
games, so no later target is leaked. This is an offline ranking evaluation over
an existing private bank, not an A/B test. Full interpretation and limitations
are in MODEL_JUSTIFICATION.md.
"""
from __future__ import annotations

import json
import random
import tempfile
from collections import Counter
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.config import MODELS_DIR, RAW_DATA_DIR, settings
from src.data.blunder_detector import clock_to_bucket, rating_to_band
from src.models.scenario_network import ScenarioNetwork
from src.training.dataset import (
    CandidateBankManager,
    HistoryScenarioDataset,
    build_training_pairs_from_bank,
)
from src.training.train import train_scenario_model

USERS = ("bonc_1", "1tshassan")
SEED = 42


def load_model(path: Path) -> ScenarioNetwork:
    model = ScenarioNetwork(
        board_embedding_dim=settings.EMBEDDING_DIM,
        history_hidden_dim=settings.HISTORY_HIDDEN_DIM,
    )
    if path.exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def rule_index(bank: list[dict[str, Any]], history: list[dict[str, Any]]) -> int:
    """Hand-written heuristic: match the last blunder's time/opponent band."""
    if not history:
        return len(bank) - 1
    last = history[-1]
    want_clock = clock_to_bucket(float(last.get("clock_remaining", 300.0)))
    want_band = rating_to_band(
        int(last.get("opponent_rating", last.get("player_rating", 1200)) or 1200)
    )
    for cand in bank:
        if (
            cand.get("clock_bucket") == want_clock
            and cand.get("opponent_rating_band") == want_band
        ):
            return bank.index(cand)
    for cand in bank:
        if cand.get("clock_bucket") == want_clock:
            return bank.index(cand)
    return len(bank) - 1


def evaluate_user(username: str) -> dict[str, Any]:
    raw_path = RAW_DATA_DIR / f"{username}_games.json"
    bank_mgr = CandidateBankManager.for_user(username)
    bank = bank_mgr.load_candidates()
    if not raw_path.exists() or len(bank) < 2:
        return {}

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    games = json.loads(raw_path.read_text(encoding="utf-8"))
    pairs = build_training_pairs_from_bank(games, username, bank)
    if len(pairs) < 6:
        return {}
    split = int(len(pairs) * 0.8)
    train, test = pairs[:split], pairs[split:]

    # Simulate a real deployment: fine-tune only on the earlier games, then
    # score the later games. Using the production checkpoint here would leak
    # the test labels because it was trained on every imported game.
    global_checkpoint = MODELS_DIR / "scenario_model.pt"
    with tempfile.TemporaryDirectory() as tmp:
        eval_checkpoint = Path(tmp) / f"{username}_eval.pt"
        training = train_scenario_model(
            train,
            bank,
            epochs=5,
            batch_size=min(8, max(1, len(train))),
            lr=1e-3,
            save_path=eval_checkpoint,
            bank_mgr=CandidateBankManager.for_user(username),
            init_from=global_checkpoint if global_checkpoint.exists() else None,
        )
        private_model = load_model(eval_checkpoint)
    global_model = load_model(global_checkpoint)

    results: dict[str, list[float]] = {
        "private_top1": [], "private_top5": [], "global_top1": [],
        "random_top1": [], "recent_top1": [], "rule_top1": [],
        "majority_category": [],
        "private_category": [], "global_category": [], "random_category": [],
        "recent_category": [], "rule_category": [], "private_category_top5": [],
        "private_similarity": [], "random_similarity": [], "rule_similarity": [],
        "recent_similarity": [],
    }
    latencies: list[float] = []

    private_embeddings = CandidateBankManager.for_user(username).compute_embeddings(
        private_model.board_encoder
    )
    norm_bank = F.normalize(private_embeddings, p=2, dim=-1)
    majority_reason = Counter(c.get("reason", "") for c in bank).most_common(1)[0][0]
    majority_pick = next(
        i for i, c in enumerate(bank) if c.get("reason", "") == majority_reason
    )

    def board_similarity(pick: int, target_idx: int) -> float:
        """Cosine similarity between the served FEN and the later real blunder FEN."""
        return float(torch.dot(norm_bank[pick], norm_bank[target_idx]).item())

    rng = random.Random(SEED)
    for model, key_prefix in ((private_model, "private"), (global_model, "global")):
        emb_mgr = CandidateBankManager.for_user(username)
        embeddings = emb_mgr.compute_embeddings(model.board_encoder)
        dataset = HistoryScenarioDataset(test, board_encoder=model.board_encoder)
        for i, item in enumerate(dataset):
            history = item["history"].unsqueeze(0)
            start = time.perf_counter()
            with torch.no_grad():
                out = model(history, embeddings)
            if key_prefix == "private":
                latencies.append((time.perf_counter() - start) * 1000)
            logits = out["retrieval_logits"][0]
            target = item["target_candidate_idx"].item()
            top5 = torch.topk(logits, min(5, len(bank))).indices.tolist()
            pred = int(top5[0])
            target_reason = bank[target].get("reason", "")

            results[f"{key_prefix}_top1"].append(float(pred == target))
            if key_prefix == "private":
                results["private_top5"].append(float(target in top5))
                results["private_category_top5"].append(
                    float(any(bank[j].get("reason", "") == target_reason for j in top5))
                )
                results["private_similarity"].append(board_similarity(pred, target))
            results[f"{key_prefix}_category"].append(
                float(bank[pred].get("reason", "") == target_reason)
            )

            if key_prefix == "private":
                # Baselines use the same target and bank; compute once here.
                random_pick = rng.randrange(len(bank))
                recent_pick = len(bank) - 1
                rule_pick = rule_index(bank, test[i].get("history", []))
                results["random_top1"].append(float(random_pick == target))
                results["recent_top1"].append(float(recent_pick == target))
                results["rule_top1"].append(float(rule_pick == target))
                results["random_category"].append(
                    float(bank[random_pick].get("reason", "") == target_reason)
                )
                results["recent_category"].append(
                    float(bank[recent_pick].get("reason", "") == target_reason)
                )
                results["rule_category"].append(
                    float(bank[rule_pick].get("reason", "") == target_reason)
                )
                results["majority_category"].append(
                    float(bank[majority_pick].get("reason", "") == target_reason)
                )
                results["random_similarity"].append(board_similarity(random_pick, target))
                results["recent_similarity"].append(board_similarity(recent_pick, target))
                results["rule_similarity"].append(board_similarity(rule_pick, target))

    # Negative control: shuffle which target belongs to which history.
    labels = [i["target_candidate_idx"].item() for i in HistoryScenarioDataset(test)]
    shuffled = labels[:]
    random.Random(SEED + 1).shuffle(shuffled)
    emb_mgr = CandidateBankManager.for_user(username)
    embeddings = emb_mgr.compute_embeddings(private_model.board_encoder)
    dataset = HistoryScenarioDataset(test, board_encoder=private_model.board_encoder)
    preds = []
    for item in dataset:
        with torch.no_grad():
            logits = private_model(item["history"].unsqueeze(0), embeddings)["retrieval_logits"][0]
        preds.append(int(torch.argmax(logits).item()))
    perm_top1 = [float(p == t) for p, t in zip(preds, shuffled)]
    perm_cat = [
        float(bank[p].get("reason", "") == bank[t].get("reason", ""))
        for p, t in zip(preds, shuffled)
    ]

    means = {k: round(float(np.mean(v)), 4) for k, v in results.items()}
    means["permutation_top1"] = round(float(np.mean(perm_top1)), 4)
    means["permutation_category"] = round(float(np.mean(perm_cat)), 4)
    means["private_train_final_loss"] = round(float(training["final_loss"]), 4)
    means["avg_inference_latency_ms"] = round(float(np.mean(latencies)), 2)
    means["test_pairs"] = len(test)
    means["bank_size"] = len(bank)
    return means


def bootstrap_ci(a: list[float], b: list[float], n: int = 2000) -> tuple[float, float]:
    """95% CI for mean(a) - mean(b) via paired bootstrap."""
    if len(a) != len(b) or not a:
        return (0.0, 0.0)
    rng = np.random.default_rng(SEED)
    diffs = []
    idx = np.arange(len(a))
    for _ in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        diffs.append(np.mean(np.asarray(a)[sample]) - np.mean(np.asarray(b)[sample]))
    return tuple(np.round(np.percentile(diffs, [2.5, 97.5]), 4))


def main() -> None:
    per_user = {}
    for user in USERS:
        result = evaluate_user(user)
        if result:
            per_user[user] = result
            print(user, json.dumps(result, indent=2))

    if not per_user:
        print("No evaluation data available")
        return

    from src.tracking import start_run, log_params, log_metrics, log_artifact
    global_ckpt = MODELS_DIR / "scenario_model.pt"

    keys = [
        "private_top1", "global_top1", "random_top1", "recent_top1", "rule_top1",
        "private_top5", "private_category", "private_category_top5",
        "random_category", "recent_category", "rule_category", "majority_category",
        "global_category",
        "private_similarity", "random_similarity", "rule_similarity", "recent_similarity",
        "permutation_top1", "permutation_category", "private_train_final_loss",
    ]
    overall = {
        k: round(float(np.mean([u[k] for u in per_user.values() if k in u])), 4)
        for k in keys
    }
    overall["users"] = list(per_user)
    overall["per_user"] = per_user
    with start_run("evaluate_usefulness", tags={"stage": "evaluate_usefulness", "dataset": "chess_com_holdout"}) as _:
        import hashlib
        log_params({
            "users": list(per_user),
            "checkpoint_md5": hashlib.md5(global_ckpt.read_bytes()).hexdigest() if global_ckpt.exists() else "none",
            "split": "chronological_80_20",
            "seed": SEED,
        })
        log_metrics(overall)
    Path("metrics_usefulness.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print("OVERALL", json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
