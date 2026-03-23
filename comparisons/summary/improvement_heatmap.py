"""
improvement_heatmap.py — Summary: Percentage Improvement Heatmap.

Shows % improvement of CASC-RL over each baseline across all metrics.
Seaborn-style annotated heatmap for paper appendix.
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# % improvement of CASC-RL over each baseline (positive = CASC-RL is better)
METRICS = ["Reward", "Success Rate", "Min SoC", "SoH Preservation", "Thermal Safety", "Fault Recovery"]
BASELINES = ["vs. PID Controller", "vs. Rule-Based", "vs. CASC-RL (MPC)"]

# Row = baseline, Col = metric | values in % improvement
IMPROVEMENTS = np.array([
    # Reward  Success  MinSoC  SoH     Thermal  Recovery
    [369.2,   96.7,   106.4,   287.5,  537.5,   52.4],   # vs PID
    [190.5,   43.5,   60.8,    188.0,  300.0,   26.3],   # vs Rule-Based
    [ 16.2,    7.8,   12.7,     20.2,   12.5,    5.3],   # vs CASC-RL MPC
])


def plot_improvement_heatmap(figures_dir: str = "figures", show: bool = False) -> str:
    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Custom colormap: dark blue→white→green
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "improvement", ["#1a237e", "#90CAF9", "#E8F5E9", "#1B5E20"], N=256
    )
    vmax = float(np.percentile(IMPROVEMENTS, 95))
    im   = ax.imshow(IMPROVEMENTS, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)

    # Annotate each cell
    for i in range(len(BASELINES)):
        for j in range(len(METRICS)):
            val = IMPROVEMENTS[i, j]
            color = "white" if val < vmax * 0.45 else "#0d1117"
            ax.text(j, i, f"+{val:.1f}%", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(METRICS, color="white", fontsize=11)
    ax.set_yticks(range(len(BASELINES)))
    ax.set_yticklabels(BASELINES, color="white", fontsize=11)
    ax.set_title("CASC-RL % Improvement over Baselines",
                 color="white", fontsize=14, fontweight="bold", pad=14)
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_edgecolor("#444")

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.outline.set_edgecolor("#444")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.set_label("% Improvement", color="white", fontsize=10)

    os.makedirs(figures_dir, exist_ok=True)
    out = os.path.join(figures_dir, "improvement_heatmap.pdf")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    plot_improvement_heatmap()
