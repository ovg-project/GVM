#!/usr/bin/env python3
from pathlib import Path
from typing import List, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# =========================
# ---- Input Metrics ------
# =========================
VLLM_P99_TTFT_MS_NO_MEM_PRESSURE = [
    246.42,
    2551.86,
    253.41,
]  # [exclusive, default-share, xsched]
VLLM_P99_ITL_MS_NO_MEM_PRESSURE = [
    64.89,
    174.60,
    68.19,
]  # [exclusive, default-share, xsched]
DIFFUSION_THROUGHPUT_NO_MEM_PRESSURE = [
    13.92,
    7.4,
    3.26,
]  # [exclusive, default-share, xsched]

VLLM_P99_TTFT_CASE_MEM_PRESSURE = [2154683.33, 9407.46]
VLLM_P99_ITL_CASE_MEM_PRESSURE = [8747.15, 994.65]
DIFFUSION_TPUT_CASE_MEM_PRESSURE = [2.38, 1.17]

# =========================
# ---- Memory Plot I/O ----
# =========================
CSV_PATH = Path("memory_current_log.csv")
PLOT_TMAX = 175
STAGES_S: List[Tuple[float, float]] = [(0, 26), (26, 50), (50, 175)]
STAGE_COLORS = ["#b3d9e6", "#ffd699", "#99cc99"]

# =========================
# ---- Styling ------------
# =========================
LABELPAD = 10
YLABEL_XCOORD_BAR = -0.42
YLABEL_XCOORD_MEM = -0.13

DEFAULT_COLOR = "#F58518"  # NVIDIA default
XSCHED_COLOR = "#54A24B"  # XSched
BAR_EDGE_LINEWIDTH = 1.8
HATCH_DEFAULT = "\\"
HATCH_XSCHED = "/"
BAR_OUTLINE_EFFECT = [pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()]

plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "hatch.linewidth": 0.8,
        "figure.constrained_layout.use": False,
    }
)


# =========================
# ---- Helpers ------------
# =========================
def _style_bar(b, edgecolor, hatch):
    b.set_hatch(hatch)
    b.set_path_effects(BAR_OUTLINE_EFFECT)
    b.set_edgecolor(edgecolor)
    b.set_linewidth(BAR_EDGE_LINEWIDTH)
    b.set_facecolor("white")


def annotate_default_bar_capped(ax, bar, value):
    """
    Put a readable label for the 'Default (with mem pressure)' bar even when it
    exceeds the y-axis cap.
    """
    ylo, yhi = ax.get_ylim()
    is_log = ax.get_yscale() == "log"

    if value >= yhi:
        text = f"{int(value / 1000):}s"
        y_text = yhi * (1.90 if is_log else 0.90)
        y_arrow = yhi * (1.00 if is_log else 0.96)
    else:
        text = f"{value:,.0f} ms"
        y_text = value * (1.05 if is_log else 1.02)
        y_arrow = value

    x_center = bar.get_x() + bar.get_width() / 2.0
    ax.annotate(
        text,
        xy=(x_center, y_arrow),
        xytext=(x_center, y_text),
        ha="center",
        va="bottom",
        fontsize=20,
        arrowprops=dict(arrowstyle="-", lw=0.8) if value >= yhi else None,
        clip_on=False,
        zorder=10,
    )


def bar_with_exclusive_line(ax, data, ylabel, log_scale=False):
    """
    For the first triplet (no pressure): show two bars (Default, XSched),
    with a red dotted exclusive baseline.
    """
    assert len(data) >= 3
    exclusive_val = data[0]
    default_val = data[1]
    xsched_val = data[2]

    x = np.arange(2)
    bars = ax.bar(x, [default_val, xsched_val], width=0.7, color="white")
    _style_bar(bars[0], DEFAULT_COLOR, HATCH_DEFAULT)
    _style_bar(bars[1], XSCHED_COLOR, HATCH_XSCHED)

    ax.axhline(y=exclusive_val, color="tab:red", linestyle="--", linewidth=3)
    ax.set_ylabel(ylabel, labelpad=LABELPAD)
    ax.yaxis.set_label_coords(YLABEL_XCOORD_BAR, 0.5)
    ax.set_xticks([])
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    if log_scale:
        ax.set_yscale("log")


def bar_no_exclusive_line(ax, data_norm, ylabel):
    """
    For normalized diffusion throughput (no red line): two bars (Default, XSched).
    `data_norm` is already normalized against exclusive.
    """
    assert len(data_norm) == 2
    x = np.arange(2)
    bars = ax.bar(x, data_norm, width=0.7, color="white")
    _style_bar(bars[0], DEFAULT_COLOR, HATCH_DEFAULT)
    _style_bar(bars[1], XSCHED_COLOR, HATCH_XSCHED)

    ax.set_ylabel(ylabel, labelpad=LABELPAD)
    ax.yaxis.set_label_coords(YLABEL_XCOORD_BAR, 0.5)
    ax.set_xticks([])
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)


def bar_two_case_with_baseline(ax, data_pair, ylabel, baseline=None, log_scale=False):
    """
    For pressure triplet (mem pressure cases): two bars (Default, XSched) with optional exclusive baseline.
    """
    assert len(data_pair) == 2
    x = np.arange(2)
    bars = ax.bar(x, data_pair, width=0.7, color="white")
    _style_bar(bars[0], DEFAULT_COLOR, HATCH_DEFAULT)
    _style_bar(bars[1], XSCHED_COLOR, HATCH_XSCHED)

    if baseline is not None:
        ax.axhline(y=baseline, color="tab:red", linestyle="--", linewidth=3)

    ax.set_ylabel(ylabel, labelpad=LABELPAD)
    ax.yaxis.set_label_coords(YLABEL_XCOORD_BAR, 0.5)
    ax.set_xticks([])
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    if log_scale:
        ax.set_yscale("log")
    return bars


def load_memory_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    t0 = df["timestamp"].dropna().iloc[0]
    df["time_s"] = (df["timestamp"] - t0).dt.total_seconds()
    df["memory_gb"] = pd.to_numeric(df["memory_current_bytes"], errors="coerce") / (
        1024**3
    )
    return df


def plot_memory(ax):
    df = load_memory_df(CSV_PATH)
    dfp = df[df["time_s"] <= PLOT_TMAX]
    engine = dfp[dfp["cmd"].str.contains("VLLM::EngineCore", na=False)]
    diff = dfp[dfp["cmd"].str.contains("diffusion.py", na=False)]

    for i, (start, end) in enumerate(STAGES_S, start=1):
        ax.axvspan(
            start, end, facecolor=STAGE_COLORS[(i - 1) % len(STAGE_COLORS)], alpha=0.35
        )
        ax.axvline(start, linestyle="--", alpha=0.35)
        ax.axvline(end, linestyle="--", alpha=0.35)
        ax.text(
            (start + end) / 2,
            0.97,
            f"#{i}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=15,
            fontweight="bold",
            color="black",
        )

    ax.plot(engine["time_s"], engine["memory_gb"], label="vLLM", linewidth=1.5)
    ax.plot(diff["time_s"], diff["memory_gb"], label="Diffusion", linewidth=1.5)
    ymax = max(engine["memory_gb"].max(), diff["memory_gb"].max())
    ax.set_ylim(0, max(45, ymax * 1.2))
    ax.axhline(40, color="red", linestyle="--", linewidth=1.5, label="Hardware Limit")
    ax.set_xlim(0, PLOT_TMAX)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GPU Memory Usage (GB)", labelpad=LABELPAD)
    ax.yaxis.set_label_coords(YLABEL_XCOORD_MEM, 0.5)
    ax.legend(
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        frameon=False,
        fontsize=15,
        borderaxespad=0.3,
    )
    ax.grid(True, linestyle="--", alpha=0.6)


def _add_group_captions(fig, group1_axes, group2_axes):
    """
    Add centered captions below the first three panels and the next three.
    Computes the mid x of each group from axes positions for robust placement.
    """

    def mid_x(ax_list):
        boxes = [ax.get_position() for ax in ax_list]
        left = min(b.x0 for b in boxes)
        right = max(b.x1 for b in boxes)
        return (left + right) / 2.0

    # Compute a shared y position just below all panels
    y_bottom = min(ax.get_position().y0 for ax in group1_axes + group2_axes)
    y_text = y_bottom - 0.08  # move slightly below axes

    fig.text(
        mid_x(group1_axes),
        y_text,
        "GPU sharing with sufficient memory",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        mid_x(group2_axes),
        y_text,
        "GPU sharing under memory pressure",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
    )


# =========================
# ---- Main ---------------
# =========================
def main():
    fig = plt.figure(figsize=(24, 4.2))
    gs = fig.add_gridspec(
        1,
        13,
        width_ratios=[1, 0.70, 1, 0.70, 1, 0.70, 1, 0.70, 1, 0.70, 1, 0.70, 3.0],
        wspace=0.0,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in [0, 2, 4, 6, 8, 10, 12]]
    ax_ttft, ax_itl, ax_tput, ax_ttft45, ax_itl45, ax_tput45, ax_mem = axes

    # ----- (no-pressure triplet) -----
    bar_with_exclusive_line(
        ax_ttft, VLLM_P99_TTFT_MS_NO_MEM_PRESSURE, "vLLM P99 TTFT (ms)", log_scale=True
    )
    ax_ttft.set_ylim(1e0, 1e4)

    bar_with_exclusive_line(
        ax_itl, VLLM_P99_ITL_MS_NO_MEM_PRESSURE, "vLLM P99 ITL (ms)"
    )

    # Normalize diffusion throughput vs exclusive (no red line)
    norm_tput = [
        x / DIFFUSION_THROUGHPUT_NO_MEM_PRESSURE[0]
        for x in DIFFUSION_THROUGHPUT_NO_MEM_PRESSURE[1:]
    ]  # [default, xsched]
    norm_tput45 = [
        x / DIFFUSION_THROUGHPUT_NO_MEM_PRESSURE[0]
        for x in DIFFUSION_TPUT_CASE_MEM_PRESSURE
    ]  # [default, xsched]
    shared_ymax = max(max(norm_tput), max(norm_tput45)) * 1.25

    bar_no_exclusive_line(ax_tput, norm_tput, "Diffusion Norm. Tput")
    ax_tput.set_ylim(0, shared_ymax)

    # ----- (pressure triplet) -----
    bars_d = bar_two_case_with_baseline(
        ax_ttft45,
        VLLM_P99_TTFT_CASE_MEM_PRESSURE,
        "vLLM P99 TTFT (ms)",
        baseline=VLLM_P99_TTFT_MS_NO_MEM_PRESSURE[0],
        log_scale=True,
    )
    ax_ttft45.set_ylim(1e0, 1e6)
    annotate_default_bar_capped(
        ax_ttft45, bars_d[0], VLLM_P99_TTFT_CASE_MEM_PRESSURE[0]
    )

    bar_two_case_with_baseline(
        ax_itl45,
        VLLM_P99_ITL_CASE_MEM_PRESSURE,
        "vLLM P99 ITL (ms)",
        baseline=VLLM_P99_ITL_MS_NO_MEM_PRESSURE[0],
        log_scale=True,
    )
    ax_itl45.set_ylim(1e0, 1e4)

    bars_f = ax_tput45.bar(np.arange(2), norm_tput45, width=0.7, color="white")
    _style_bar(bars_f[0], DEFAULT_COLOR, HATCH_DEFAULT)
    _style_bar(bars_f[1], XSCHED_COLOR, HATCH_XSCHED)
    ax_tput45.set_ylabel("Diffusion Norm. Tput", labelpad=LABELPAD)
    ax_tput45.yaxis.set_label_coords(YLABEL_XCOORD_BAR, 0.5)
    ax_tput45.set_xticks([])
    ax_tput45.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_tput45.set_ylim(0, shared_ymax)

    if CSV_PATH.exists():
        plot_memory(ax_mem)
    else:
        ax_mem.text(
            0.5, 0.5, f"Missing {CSV_PATH.name}", ha="center", va="center", fontsize=20
        )
        ax_mem.set_axis_off()

    default_patch = Rectangle(
        (0, 0),
        1,
        1,
        facecolor="white",
        edgecolor=DEFAULT_COLOR,
        linewidth=BAR_EDGE_LINEWIDTH,
        hatch=HATCH_DEFAULT,
        path_effects=BAR_OUTLINE_EFFECT,
    )
    xsched_patch = Rectangle(
        (0, 0),
        1,
        1,
        facecolor="white",
        edgecolor=XSCHED_COLOR,
        linewidth=BAR_EDGE_LINEWIDTH,
        hatch=HATCH_XSCHED,
        path_effects=BAR_OUTLINE_EFFECT,
    )
    exclusive_handle = Line2D([], [], color="tab:red", linestyle="--", linewidth=3)

    fig.legend(
        [default_patch, xsched_patch, exclusive_handle],
        ["NVIDIA default", "XSched", "Exclusive"],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
        fontsize=20,
    )

    _add_group_captions(
        fig, [ax_ttft, ax_itl, ax_tput], [ax_ttft45, ax_itl45, ax_tput45]
    )

    plt.subplots_adjust(left=0.085, right=0.985, bottom=0.11, top=0.80)

    out = Path("motiv_mem_perf_study.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
