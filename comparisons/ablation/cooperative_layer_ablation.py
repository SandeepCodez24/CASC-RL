"""
cooperative_layer_ablation.py — Ablation: Cooperative MARL vs. Independent PPO.
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
    "CASC-RL (MAPPO)":   {"color": "#2196F3", "reward": (1.22, 0.08), "coop": (0.72, 0.05)},
    "Independent PPO":   {"color": "#FF9800", "reward": (0.79, 0.12), "coop": (0.42, 0.08)},
    "Shared Policy":     {"color": "#4CAF50", "reward": (0.91, 0.10), "coop": (0.55, 0.07)},
    "No Reward Shaping": {"color": "#9C27B0", "reward": (0.98, 0.11), "coop": (0.61, 0.07)},
}

def plot_cooperative_ablation(figures_dir="figures", show=False):
    names  = list(VARIANTS.keys())
    colors = [VARIANTS[n]["color"] for n in names]
    x      = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes: ax.set_facecolor("#0f1117")

    for ax, key, ylabel, title in [
        (axes[0], "reward", "Mean Episode Reward", "Cooperative vs. Independent: Reward"),
        (axes[1], "coop",   "Coordination Efficiency", "Cooperative vs. Independent: Efficiency"),
    ]:
        vals = [VARIANTS[n][key][0] for n in names]
        errs = [VARIANTS[n][key][1] for n in names]
        bars = ax.bar(x, vals, yerr=errs, color=colors, alpha=0.85, width=0.5,
                      error_kw={"ecolor": "white", "capsize": 5})
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(errs)*0.1,
                    f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" (", "\n(") for n in names], color="white", fontsize=9)
        ax.set_ylabel(ylabel, color="white", fontsize=11)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_edgecolor("#444")
        ax.grid(axis="y", alpha=0.15, color="white", linestyle="--")

    fig.suptitle("Ablation: Cooperative MARL Layer", color="white", fontsize=14, fontweight="bold")
    os.makedirs(figures_dir, exist_ok=True)
    out = os.path.join(figures_dir, "cooperative_layer_ablation.pdf")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show: plt.show()
    plt.close(fig); print(f"Saved: {out}"); return out

if __name__ == "__main__":
    plot_cooperative_ablation()
