"""
hierarchical_layer_ablation.py — Ablation: With vs. Without Hierarchical Coordinator (Layer 4).
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

VARIANTS = {
    "With Coordinator": {"color": "#2196F3", "reward": 1.22, "tasks": 3.2,  "conflict": 0.04},
    "No Coordinator":   {"color": "#FF9800", "reward": 0.98, "tasks": 2.5,  "conflict": 0.14},
    "Greedy Assign.":   {"color": "#4CAF50", "reward": 1.08, "tasks": 2.9,  "conflict": 0.08},
    "Random Assign.":   {"color": "#F44336", "reward": 0.71, "tasks": 1.8,  "conflict": 0.23},
}
METRIC_LABELS = {"reward": "Mean Episode Reward", "tasks": "Tasks/Orbit", "conflict": "Conflict Rate"}

def plot_hierarchical_ablation(figures_dir="figures", show=False):
    names  = list(VARIANTS.keys())
    colors = [VARIANTS[n]["color"] for n in names]
    x      = np.arange(len(names))
    metrics = ["reward", "tasks", "conflict"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes: ax.set_facecolor("#0f1117")

    for ax, metric in zip(axes, metrics):
        vals = [VARIANTS[n][metric] for n in names]
        bars = ax.bar(x, vals, color=colors, alpha=0.85, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    f"{v:.2f}", ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], color="white", fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric], color="white", fontsize=11)
        ax.set_title(f"Ablation: {METRIC_LABELS[metric]}", color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_edgecolor("#444")
        ax.grid(axis="y", alpha=0.15, color="white", linestyle="--")

    fig.suptitle("Ablation: Hierarchical Coordination Layer (Layer 4)",
                 color="white", fontsize=14, fontweight="bold")
    os.makedirs(figures_dir, exist_ok=True)
    out = os.path.join(figures_dir, "hierarchical_layer_ablation.pdf")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show: plt.show()
    plt.close(fig); print(f"Saved: {out}"); return out

if __name__ == "__main__":
    plot_hierarchical_ablation()
