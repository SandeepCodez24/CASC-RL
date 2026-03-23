"""
radar_chart_all_methods.py — Summary: Radar/Spider Chart of All Metrics.

Plots a multi-axis radar chart comparing all 4 algorithms across 6 metrics
simultaneously. Classic "spider web" Figure for multi-criteria comparison papers.
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Normalized scores [0,1]: 1=best, 0=worst  (all metrics aligned: higher = better)
CATEGORIES = ["Reward", "Mission\nSuccess", "Min SoC", "SoH\nPreservation", "Thermal\nSafety", "Fault\nRecovery"]
ALGO_DATA = {
    "CASC-RL (Ours)":    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "CASC-RL (MPC)":     [0.86, 0.93, 0.89, 0.87, 0.93, 0.95],
    "Rule-Based":        [0.34, 0.70, 0.62, 0.43, 0.55, 0.76],
    "PID Controller":    [0.21, 0.51, 0.48, 0.28, 0.35, 0.63],
}
ALGO_COLORS = {
    "CASC-RL (Ours)": "#2196F3",
    "CASC-RL (MPC)":  "#4CAF50",
    "Rule-Based":     "#FF9800",
    "PID Controller": "#F44336",
}


def plot_radar(figures_dir: str = "figures", show: bool = False) -> str:
    N    = len(CATEGORIES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig = plt.figure(figsize=(9, 8))
    fig.patch.set_facecolor("#0f1117")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#0d1117")

    # Draw grid
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CATEGORIES, color="white", fontsize=11)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="#888", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(color="#333", linestyle="--", linewidth=0.8)
    ax.spines["polar"].set_color("#555")

    for algo, values in ALGO_DATA.items():
        vals    = values + values[:1]
        color   = ALGO_COLORS[algo]
        lw      = 3.0 if "Ours" in algo else 1.8
        alpha_f = 0.20 if "Ours" in algo else 0.07
        ax.plot(angles, vals,  color=color, linewidth=lw, linestyle="solid", label=algo)
        ax.fill(angles, vals,  color=color, alpha=alpha_f)
        # Marker at each vertex
        ax.scatter(angles, vals, color=color, s=40 if "Ours" in algo else 20, zorder=5)

    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.15),
              facecolor="#1a1d23", edgecolor="#555", labelcolor="white", fontsize=10)
    ax.set_title("Multi-Metric Spider Chart — 3 Satellite Constellation",
                 color="white", fontsize=13, fontweight="bold", pad=22)

    os.makedirs(figures_dir, exist_ok=True)
    out = os.path.join(figures_dir, "radar_chart_all_methods.pdf")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    plot_radar()
