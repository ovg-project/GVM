#!/usr/bin/env python3
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# =========================
# Global Configuration
# =========================
FONT_SIZE = 15

# Color Configuration
DEFAULT_COLOR = "#F58518"  # NVIDIA default
XSCHED_COLOR = "tab:green"  # XSched
EXCLUSIVE_LINE_COLOR = "tab:red"
EXCLUSIVE_LINE_STYLE = "--"
EXCLUSIVE_LINE_WIDTH = 3

# Bar Styling
BAR_EDGE_LINEWIDTH = 2.5
BAR_WIDTH = 0.3
HATCH_DEFAULT = "\\"
HATCH_XSCHED = "/"

# Input Metrics
# Format: [exclusive, default-share, xsched]
VLLM_P99_TTFT_MS_NO_MEM_PRESSURE = [246.42, 2551.86, 253.41]
VLLM_P99_ITL_MS_NO_MEM_PRESSURE = [64.89, 174.60, 68.19]
DIFFUSION_THROUGHPUT_NO_MEM_PRESSURE = [13.92, 7.4, 3.26]

# Format: [default, xsched]
VLLM_P99_TTFT_CASE_MEM_PRESSURE = [2154683.33, 9407.46]
VLLM_P99_ITL_CASE_MEM_PRESSURE = [8747.15, 994.65]
DIFFUSION_TPUT_CASE_MEM_PRESSURE = [2.38, 1.17]

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
def style_bar(bar, edgecolor, hatch):
    """Apply consistent styling to a bar."""
    bar.set_hatch(hatch)
    bar.set_edgecolor(edgecolor)
    bar.set_linewidth(BAR_EDGE_LINEWIDTH)
    bar.set_facecolor("white")


def create_bar_plot(ax, values, ylabel, exclusive_val=None, log_scale=False):
    """Create bar plot with optional exclusive baseline line."""
    x = np.array([0, 0.5])  # Closer spacing between bars
    bars = ax.bar(x, values, width=BAR_WIDTH, color="white", edgecolor="black", linewidth=1.5)
    style_bar(bars[0], DEFAULT_COLOR, HATCH_DEFAULT)
    style_bar(bars[1], XSCHED_COLOR, HATCH_XSCHED)

    if exclusive_val is not None:
        ax.axhline(
            y=exclusive_val,
            color=EXCLUSIVE_LINE_COLOR,
            linestyle=EXCLUSIVE_LINE_STYLE,
            linewidth=EXCLUSIVE_LINE_WIDTH,
        )

    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    if log_scale:
        ax.set_yscale("log")
        min_val = min(values + ([exclusive_val] if exclusive_val is not None else []))
        ax.set_ylim(bottom=min_val * 0.1)
    else:
        ax.set_ylim(bottom=0)
    return bars


def annotate_capped_bar(ax, bar, value, y_limit):
    """Add text label for bar that exceeds y-axis cap."""
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y_limit,
        f"{value:.0f}",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE - 1,
        color="tab:red",
    )


def add_group_captions(fig, group1_axes, group2_axes):
    """Add centered captions below panel groups. Returns caption positions."""
    def mid_x(ax_list):
        boxes = [ax.get_position() for ax in ax_list]
        return (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2.0

    # Top group caption (below first group, outside axes)
    y_text_top = min(ax.get_position().y0 for ax in group1_axes) - 0.02

    # Bottom group caption (below second group, outside axes)
    y_text_bottom = min(ax.get_position().y0 for ax in group2_axes) - 0.02

    # Top group: "With sufficient memory"
    fig.text(
        mid_x(group1_axes), y_text_top, "(a) With sufficient memory",
        ha="center", va="top", fontsize=FONT_SIZE, fontweight="bold",
        transform=fig.transFigure
    )

    # Bottom group: "Under memory pressure"
    fig.text(
        mid_x(group2_axes), y_text_bottom, "(b) Under memory pressure",
        ha="center", va="top", fontsize=FONT_SIZE, fontweight="bold",
        transform=fig.transFigure
    )

    return y_text_top, y_text_bottom


def create_legend_handles():
    """Create legend handles matching bar appearance."""
    default_patch = mpatches.Patch(
        facecolor="white",
        edgecolor=DEFAULT_COLOR,
        linewidth=BAR_EDGE_LINEWIDTH,
        hatch=HATCH_DEFAULT,
        label="NVIDIA default",
    )
    xsched_patch = mpatches.Patch(
        facecolor="white",
        edgecolor=XSCHED_COLOR,
        linewidth=BAR_EDGE_LINEWIDTH,
        hatch=HATCH_XSCHED,
        label="XSched",
    )
    exclusive_handle = Line2D(
        [], [],
        color=EXCLUSIVE_LINE_COLOR,
        linestyle=EXCLUSIVE_LINE_STYLE,
        linewidth=EXCLUSIVE_LINE_WIDTH,
        label="Exclusive",
    )
    return [default_patch, xsched_patch, exclusive_handle]


# =========================
# Main
# =========================
def main():
    # Convert TTFT from ms to seconds
    TTFT_NO_PRESSURE_S = [x / 1000.0 for x in VLLM_P99_TTFT_MS_NO_MEM_PRESSURE]
    TTFT_PRESSURE_S = [x / 1000.0 for x in VLLM_P99_TTFT_CASE_MEM_PRESSURE]

    # Create figure with 2x5 grid
    fig = plt.figure(figsize=(6, 5))
    gs = fig.add_gridspec(
        2, 5,
        width_ratios=[0.4, 0.35, 0.4, 0.65, 0.4],
        height_ratios=[1, 1],
        wspace=0.0,
        hspace=0.6,
    )

    # Row 1: No memory pressure
    ax_ttft = fig.add_subplot(gs[0, 0])
    ax_itl = fig.add_subplot(gs[0, 2])
    ax_tput = fig.add_subplot(gs[0, 4])

    # Row 2: Memory pressure
    ax_ttft45 = fig.add_subplot(gs[1, 0])
    ax_itl45 = fig.add_subplot(gs[1, 2])
    ax_tput45 = fig.add_subplot(gs[1, 4])

    # Normalize diffusion throughput
    norm_tput = [
        x / DIFFUSION_THROUGHPUT_NO_MEM_PRESSURE[0]
        for x in DIFFUSION_THROUGHPUT_NO_MEM_PRESSURE[1:]
    ]
    norm_tput45 = [
        x / DIFFUSION_THROUGHPUT_NO_MEM_PRESSURE[0]
        for x in DIFFUSION_TPUT_CASE_MEM_PRESSURE
    ]
    shared_ymax = max(max(norm_tput), max(norm_tput45)) * 1.25

    # No memory pressure plots
    create_bar_plot(
        ax_ttft, TTFT_NO_PRESSURE_S[1:], "P99 TTFT (s)",
        exclusive_val=TTFT_NO_PRESSURE_S[0], log_scale=True,
    )
    ax_ttft.set_ylim(bottom=min(TTFT_NO_PRESSURE_S) * 0.1)

    create_bar_plot(
        ax_itl, VLLM_P99_ITL_MS_NO_MEM_PRESSURE[1:], "P99 ITL (ms)",
        exclusive_val=VLLM_P99_ITL_MS_NO_MEM_PRESSURE[0],
    )

    create_bar_plot(ax_tput, norm_tput, "Norm. Tput. (x)")
    ax_tput.set_ylim(0, shared_ymax)

    # Memory pressure plots
    bars_ttft = create_bar_plot(
        ax_ttft45, TTFT_PRESSURE_S, "P99 TTFT (s)",
        exclusive_val=TTFT_NO_PRESSURE_S[0], log_scale=True,
    )
    ttft_all = TTFT_PRESSURE_S + [TTFT_NO_PRESSURE_S[0]]
    ax_ttft45.set_ylim(bottom=min(ttft_all) * 0.1, top=max(ttft_all) * 10)
    if TTFT_PRESSURE_S[0] > ax_ttft45.get_ylim()[1]:
        annotate_capped_bar(ax_ttft45, bars_ttft[0], TTFT_PRESSURE_S[0], ax_ttft45.get_ylim()[1])

    create_bar_plot(
        ax_itl45, VLLM_P99_ITL_CASE_MEM_PRESSURE, "P99 ITL (ms)",
        exclusive_val=VLLM_P99_ITL_MS_NO_MEM_PRESSURE[0], log_scale=True,
    )
    ax_itl45.set_ylim(1e0, 1e4)

    create_bar_plot(ax_tput45, norm_tput45, "Norm. Tput. (x)")
    ax_tput45.set_ylim(0, shared_ymax)

    # Add vertical separators between vLLM and diffusion groups
    for row_axes in [[ax_ttft, ax_itl, ax_tput], [ax_ttft45, ax_itl45, ax_tput45]]:
        ax_vllm_last = row_axes[1]
        ax_diff_first = row_axes[2]
        vllm_pos = ax_vllm_last.get_position()
        separator_x = (vllm_pos.x1 + ax_diff_first.get_position().x0) / 2 - 0.025
        # Move vertical separator upward
        y_offset = 0.07
        fig.add_artist(Line2D(
            [separator_x, separator_x],
            [vllm_pos.y0 + y_offset - 0.04, vllm_pos.y1 + y_offset + 0.04],
            color="gray", linewidth=2, linestyle="--", alpha=0.6,
            transform=fig.transFigure, zorder=0,
        ))

    # Group captions
    caption_y_top, caption_y_bottom = add_group_captions(
        fig, [ax_ttft, ax_itl, ax_tput], [ax_ttft45, ax_itl45, ax_tput45]
    )

    # # Horizontal separator between groups
    # top_group_bottom = min(ax.get_position().y0 for ax in [ax_ttft, ax_itl, ax_tput])
    # bottom_group_top = max(ax.get_position().y1 for ax in [ax_ttft45, ax_itl45, ax_tput45])
    # separator_y = (top_group_bottom + bottom_group_top) / 2 + 0.05  # Move upward

    # left_x = min(ax.get_position().x0 for ax in [ax_ttft, ax_itl, ax_tput]) - 0.18
    # right_x = max(ax.get_position().x1 for ax in [ax_ttft, ax_itl, ax_tput]) + 0.09

    # fig.add_artist(Line2D(
    #     [left_x, right_x], [separator_y, separator_y],
    #     color="black", linewidth=2, linestyle="-",
    #     transform=fig.transFigure, zorder=0,
    # ))

    # Legend
    legend_handles = create_legend_handles()
    fig.legend(
        handles=legend_handles,
        loc=(0.04, 0.94),
        ncol=len(legend_handles),
        frameon=False,
    )

    # Add vLLM and Diffusion labels above group captions for both rows
    def mid_x(ax_list):
        boxes = [ax.get_position() for ax in ax_list]
        return (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2.0

    y_caption_top = caption_y_top + 0.01
    y_caption_bottom = caption_y_bottom + 0.01

    # Top row captions
    vllm_center_x_top = mid_x([ax_ttft, ax_itl])
    fig.text(
        vllm_center_x_top, y_caption_top, "vLLM Inference (↓ is better)",
        fontsize=FONT_SIZE, ha="center", va="bottom", transform=fig.transFigure
    )
    diff_center_x_top = mid_x([ax_tput])
    fig.text(
        diff_center_x_top, y_caption_top, "Diffusion (↑ is better)",
        fontsize=FONT_SIZE, ha="center", va="bottom", transform=fig.transFigure
    )

    # Bottom row captions
    vllm_center_x_bottom = mid_x([ax_ttft45, ax_itl45])
    fig.text(
        vllm_center_x_bottom, y_caption_bottom, "vLLM Inference (↓ is better)",
        fontsize=FONT_SIZE, ha="center", va="bottom", transform=fig.transFigure
    )
    diff_center_x_bottom = mid_x([ax_tput45])
    fig.text(
        diff_center_x_bottom, y_caption_bottom, "Diffusion (↑ is better)",
        fontsize=FONT_SIZE, ha="center", va="bottom", transform=fig.transFigure
    )

    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.95)
    fig.savefig("motiv_perf_study.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
