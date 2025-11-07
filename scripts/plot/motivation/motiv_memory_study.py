#!/usr/bin/env python3
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# =========================
# Global Configuration
# =========================
FONT_SIZE = 15

# Memory Plot Configuration
CSV_PATH = Path("memory_current_log.csv")
PLOT_TMAX = 160
STAGES_S: List[Tuple[float, float]] = [(0, 26), (26, 50), (50, 175)]
STAGE_COLORS = ["#b3d9e6", "#ffd699", "#99cc99"]

# Matplotlib Configuration
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "hatch.linewidth": 0.8,
    "figure.constrained_layout.use": False,
    # Type 1 font configuration for PDF submission
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
})


# =========================
# Helper Functions
# =========================
def load_memory_df(csv_path: Path) -> pd.DataFrame:
    """Load and process memory data from CSV."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    t0 = df["timestamp"].dropna().iloc[0]
    df["time_s"] = (df["timestamp"] - t0).dt.total_seconds()
    df["memory_gb"] = pd.to_numeric(df["memory_current_bytes"], errors="coerce") / (1024**3)
    return df


def plot_memory(ax):
    """Plot memory usage over time with stage annotations."""
    df = load_memory_df(CSV_PATH)
    dfp = df[df["time_s"] <= PLOT_TMAX]
    engine = dfp[dfp["cmd"].str.contains("VLLM::EngineCore", na=False)]
    diff = dfp[dfp["cmd"].str.contains("diffusion.py", na=False)]

    # Add stage annotations
    for i, (start, end) in enumerate(STAGES_S, start=1):
        ax.axvspan(start, end, facecolor=STAGE_COLORS[i - 1], alpha=0.25)
        ax.axvline(start, linestyle="--", alpha=0.35)
        ax.axvline(end, linestyle="--", alpha=0.35)
        # Position labels above the figure using axes transform with y > 1
        ax.text(
            (start + end) / 2, 1.05, f"#{i}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=FONT_SIZE,
            fontweight="bold", color="black",
        )

    ax.plot(engine["time_s"], engine["memory_gb"], label="vLLM", linewidth=1.5, marker="o", markersize=6, markevery=10, fillstyle="none", markeredgewidth=1.5)
    ax.set_ylim(0, 42)
    ax.set_yticks(range(0, 42, 10))
    ax.plot([0, 75], [40, 40], color="black", linestyle="--", linewidth=2, alpha=0.6)
    ax.text(PLOT_TMAX * 0.98, 39.5, "GPU capacity",
            ha="right", va="center", fontsize=FONT_SIZE-2,
            color="black", bbox=dict(boxstyle="round,pad=0.3", facecolor="none", edgecolor="none", alpha=0.7))
    ax.plot(diff["time_s"], diff["memory_gb"], label="Diffusion", linewidth=1.5, marker="s", markersize=6, markevery=10, fillstyle="none", markeredgewidth=1.5)
    ax.set_xlim(0, PLOT_TMAX)
    ax.set_xticks(range(0, int(PLOT_TMAX) + 1, 50))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory Usage (GB)")
    # ax.legend(
    #     loc=(-0.3, 1.15),
    #     frameon=False, fontsize=FONT_SIZE,
    #     ncol=2,
    #     columnspacing=1.3,
    # )
    ax.legend(
        loc=(0.37, 0.3),
        frameon=False, fontsize=FONT_SIZE,
        columnspacing=1.3,
    )
    ax.grid(True, linestyle="--", alpha=0.6)


# =========================
# Main
# =========================
def main():
    fig = plt.figure(figsize=(3.7, 4))
    ax = fig.add_subplot(111)

    if CSV_PATH.exists():
        plot_memory(ax)
    else:
        ax.text(0.5, 0.5, f"Missing {CSV_PATH.name}",
                ha="center", va="center", fontsize=FONT_SIZE)
        ax.set_axis_off()

    # # Add caption below the plot
    # fig.text(0.53, 0.02, "(c) Mem. capacity contention",
    #          ha="center", va="bottom", fontsize=FONT_SIZE, fontweight="bold")

    plt.tight_layout(rect=[0, 0.05, 1, 0.88])
    fig.savefig("motiv_memory_study.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
