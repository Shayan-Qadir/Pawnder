"""Generate the simple explanation figures used by MODEL_EXPLAINED.md."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
METRICS = json.loads((ROOT / "metrics_usefulness.json").read_text(encoding="utf-8"))

OUT.mkdir(parents=True, exist_ok=True)

DARK = "#161512"
PANEL = "#262421"
GREEN = "#81b64c"
AMBER = "#e8b339"
STONE = "#a8a29e"


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.figure.patch.set_facecolor(DARK)
    ax.tick_params(colors="white", labelsize=10)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#57534e")
    ax.grid(axis="y", color="#44403c", alpha=0.5, linewidth=0.8)
    ax.set_axisbelow(True)


def bar_chart(filename: str, title: str, ylabel: str, values: dict[str, float], fmt: str) -> None:
    labels = list(values)
    nums = [values[k] for k in labels]
    colors = [GREEN if k.startswith("Model") or k.startswith("Pawnder") else STONE for k in labels]
    labels_text = (
        [f"{n:.1%}" for n in nums] if fmt == "percent" else [f"{n:.3f}" for n in nums]
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    style_axes(ax)
    bars = ax.bar(labels, nums, color=colors, width=0.62)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(nums) * 1.18)
    ax.bar_label(bars, labels=labels_text, padding=4, color="white", fontsize=10)
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)


def make_flow() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor(DARK)

    boxes = [
        (0.3, 1.7, 2.0, 1.6, "INPUT 1\nLast 5 blunders\n(board + clock + rating)", GREEN),
        (0.3, 0.15, 2.0, 1.2, "INPUT 2\nYour private bank\nof past blunders", AMBER),
        (3.0, 0.9, 2.2, 1.9, "MODEL\nCNN + GRU\nlearned ranking", GREEN),
        (5.9, 0.9, 2.2, 1.9, "OUTPUT\nOne score per\nstored position", AMBER),
        (8.7, 0.9, 2.0, 1.9, "PRACTICE\nLegal FEN +\nStockfish reply", GREEN),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.5, edgecolor=color, facecolor=PANEL,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=10)

    arrows = [
        (2.35, 2.5, 2.9, 1.9),
        (2.35, 0.75, 2.9, 1.35),
        (5.25, 1.85, 5.8, 1.85),
        (8.15, 1.85, 8.6, 1.85),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=STONE, lw=2))

    ax.text(5.5, 4.35, "How Pawnder picks a drill", ha="center", va="center",
            color="white", fontsize=16, fontweight="bold")
    ax.text(5.5, 3.65, "It never invents a board — it ranks your own saved blunders.",
            ha="center", color=STONE, fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "model_flow.png", dpi=160)
    plt.close(fig)


def make_split() -> None:
    fig, ax = plt.subplots(figsize=(9, 2.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    fig.patch.set_facecolor(DARK)

    ax.add_patch(FancyBboxPatch((0.4, 0.65), 7.2, 0.75,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=GREEN, edgecolor="none"))
    ax.add_patch(FancyBboxPatch((7.6, 0.65), 2.0, 0.75,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=AMBER, edgecolor="none"))
    ax.text(4.0, 1.025, "OLDER GAMES — train on 80%", ha="center", va="center",
            color=DARK, fontsize=11, fontweight="bold")
    ax.text(8.6, 1.025, "NEW GAMES — test on 20%", ha="center", va="center",
            color=DARK, fontsize=10, fontweight="bold")
    ax.text(5.0, 1.85, "Chronological split: the model only sees the past, then we test on the future",
            ha="center", color="white", fontsize=13, fontweight="bold")
    ax.text(5.0, 0.25, "This stops the model from cheating by learning the test answers.",
            ha="center", color=STONE, fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT / "holdout_split.png", dpi=160)
    plt.close(fig)


def make_top5() -> None:
    model = METRICS["private_top5"]
    chance = 0.075  # mean of 5/51 and 5/95
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    style_axes(ax)
    bars = ax.bar(["Model", "Random chance"], [model, chance], color=[GREEN, STONE], width=0.5)
    ax.set_title("Was the real later blunder in the top 5?", fontsize=13, fontweight="bold")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 0.7)
    ax.bar_label(bars, labels=[f"{model:.1%}", f"{chance:.1%}"], padding=4, color="white")
    plt.tight_layout()
    fig.savefig(OUT / "top5_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    make_flow()
    make_split()
    bar_chart(
        "category_comparison.png",
        "Did it practise the right kind of mistake?",
        "Same-category hit rate",
        {
            "Model": METRICS["private_category"],
            "Most recent": METRICS["recent_category"],
            "Random": METRICS["random_category"],
            "Hand rule": METRICS["rule_category"],
        },
        "percent",
    )
    bar_chart(
        "similarity_comparison.png",
        "Did it pick a visually similar position?",
        "Board similarity (cosine)",
        {
            "Model": METRICS["private_similarity"],
            "Most recent": METRICS["recent_similarity"],
            "Hand rule": METRICS["rule_similarity"],
            "Random": METRICS["random_similarity"],
        },
        "float",
    )
    make_top5()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
