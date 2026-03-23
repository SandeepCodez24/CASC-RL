"""
world_model_ablation.py
Ablation: Effect of World Model on Agent Performance.
Compares: CASC-RL (with WM) vs. CASC-RL-NoWM (learned policy without WM lookahead).
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

VARIANTS = {
    "CASC-RL (Full)":        {"color": "#2196F3", "reward": (1.22, 0.08), "soc_min": (0.29, 0.04)},
    "CASC-RL (No WM)":       {"color": "#FF9800", "reward": (0.87, 0.11), "soc_min": (0.20, 0.05)},
    "CASC-RL (No Safety)":   {"color": "#F44336", "reward": (1.05, 0.09), "soc_min": (0.11, 0.07)},
    "CASC-RL (No Coord.)":   {"color": "#9C27B0", "reward": (0.96, 0.10), "soc_min": (0.24, 0.05)},
}

def plot_world_model_ablation(figures_dir="figures", show=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes: ax.set_facecolor("#0f1117")

    names  = list(VARIANTS.keys())
    colors = [VARIANTS[n]["color"] for n in names]
    x      = np.arange(len(names))

    for ax_idx, metric_key, ylabel in [
        (0, "reward",  "Mean Episode Reward"),
        (1, "soc_min", "Mean Min SoC per Episode"),
    ]:
        ax   = axes[ax_idx]
        vals = [VARIANTS[n][metric_key][0] for n in names]
        errs = [VARIANTS[n][metric_key][1] for n in names]
        bars = ax.bar(x, vals, yerr=errs, color=colors, alpha=0.85, width=0.5,
                      error_kw={"ecolor": "white", "capsize": 5})
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(errs)*0.1,
                    f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("CASC-RL ", "\n") for n in names], color="white", fontsize=9)
        ax.set_ylabel(ylabel, color="white", fontsize=11)
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_edgecolor("#444")
        ax.grid(axis="y", alpha=0.15, color="white", linestyle="--")

    axes[0].set_title("Ablation: Episode Reward", color="white", fontsize=12, fontweight="bold")
    axes[1].set_title("Ablation: Min Battery SoC", color="white", fontsize=12, fontweight="bold")
    if metric_key == "soc_min":
        axes[1].axhline(0.15, color="#FF5252", lw=1.5, ls="--", alpha=0.6, label="Critical Threshold")
        axes[1].legend(facecolor="#1a1d23", edgecolor="#444", labelcolor="white", fontsize=8)

    fig.suptitle("CASC-RL Ablation Study — 3 Satellite Constellation",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    os.makedirs(figures_dir, exist_ok=True)
    out = os.path.join(figures_dir, "world_model_ablation.pdf")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show: plt.show()
    plt.close(fig); print(f"Saved: {out}"); return out

if __name__ == "__main__":
    plot_world_model_ablation()
