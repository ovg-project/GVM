import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

# Global Configuration
FONT_SIZE = 15
FIGURE_SIZE = (14, 2.8)
METHODS_ORDER = ['TGS', 'GPreempt', 'xsched', 'GVM', 'exclusive']
HATCH_PATTERNS = ['/', '\\', 'x', 'o']
VLLM_SPACING_REDUCTION = 0.65  # Reduce vLLM subplot gaps to 65% of original
MIN_SUBPLOT_GAP = 0.015  # Minimum gap to avoid text overlap
SHOW_LEGEND = True

# Color Configuration
METHOD_COLORS = {
    'TGS': 'tab:blue',
    'GPreempt': 'tab:orange',
    'xsched': 'tab:green',
    'GVM': 'tab:purple',
    'exclusive': 'tab:red'
}
EXCLUSIVE_LINE_COLOR = 'tab:red'
EXCLUSIVE_LINE_STYLE = '--'
EXCLUSIVE_LINE_WIDTH = 3

# vLLM metrics: (source_key, display_label, scale_factor, use_log_scale)
VLLM_METRICS = [
    ('Median TTFT (ms)', 'Median TTFT (s)', 0.001, True),
    ('P99 TTFT (ms)', 'P99 TTFT (s)', 0.001, True),
    ('Median ITL (ms)', 'Median ITL (ms)', 1.0, False),
    ('P99 ITL (ms)', 'P99 ITL (ms)', 1.0, False),
]

def calculate_metric(application, filepath):
    if application == "vllm":
        with open(filepath) as f:
            data = json.load(f)
        return {
            "Median TTFT (ms)": data["median_ttft_ms"],
            "P99 TTFT (ms)": data["p99_ttft_ms"],
            "Median ITL (ms)": data["median_itl_ms"],
            "P99 ITL (ms)": data["p99_itl_ms"],
        }
    elif application == "diffusion":
        latency = np.mean(np.loadtxt(filepath))
        tput = 1 / latency
        return {"latency": latency, "tput": tput}
    elif application == "llama_factory":
        latency = np.loadtxt(filepath)
        tput = 1 / latency
        return {"latency": latency, "tput": tput}
    else:
        raise NotImplementedError(f"Application: {application} is not supported")

def load_data(input_path):
    applications = os.path.basename(os.path.normpath(input_path)).split("+")
    results = {application: {} for application in applications}

    for f in os.listdir(input_path):
        application = f.split("-")[0]
        parts = f.split("-")
        # Find the timestamp part (starts with year 2025) or use last 2 parts for timestamp
        timestamp_idx = -1
        for i, part in enumerate(parts):
            if part.startswith("2025"):
                timestamp_idx = i
                break

        if timestamp_idx > 0:
            method_full = "-".join(parts[1:timestamp_idx])
        else:
            method_full = "-".join(parts[1:-2])  # Remove last 2 parts (timestamp)

        # Normalize method name: extract base method name that matches METHODS_ORDER
        # e.g., "GVM-2-10" -> "GVM", "GPreempt" -> "GPreempt"
        method = method_full
        if method_full not in METHODS_ORDER:
            # Try to find a matching method from METHODS_ORDER
            # This handles cases like "GVM-2-10" -> "GVM"
            for m in METHODS_ORDER:
                if method_full.startswith(m):
                    method = m
                    break
            # If no match found, use the base name (first part)
            if method not in METHODS_ORDER:
                method = method_full.split("-")[0]

        results[application][method] = calculate_metric(application, os.path.join(input_path, f))

    return results, applications

def get_color_map():
    """Build color map from explicit color configuration."""
    return {method: METHOD_COLORS[method] for method in METHODS_ORDER}

def create_grouped_bar_plot(ax, values_sufficient, values_pressure, method_names, color_map, hatch_map):
    """Create grouped bar plot with two bars side-by-side for each method."""
    x = np.arange(len(method_names))
    width = 0.35  # Width of each bar, leaving space between groups

    bars_sufficient = ax.bar(x - width/2, values_sufficient, width,
                            color='white', edgecolor='black', linewidth=1.5, alpha=0.7)
    bars_pressure = ax.bar(x + width/2, values_pressure, width,
                          color='white', edgecolor='black', linewidth=1.5, alpha=0.6)

    for bar, method in zip(bars_sufficient, method_names):
        if method == 'GVM':
            rgba_color = to_rgba(color_map[method], alpha=0.7)
            bar.set_facecolor(rgba_color)
            bar.set_edgecolor('black')
        else:
            bar.set_edgecolor(color_map[method])
        bar.set_hatch(hatch_map[method])
        bar.set_linewidth(2.5)

    for bar, method in zip(bars_pressure, method_names):
        if method == 'GVM':
            rgba_color = to_rgba(color_map[method], alpha=0.6)
            bar.set_facecolor(rgba_color)
            bar.set_edgecolor('black')
        else:
            bar.set_edgecolor(color_map[method])
        bar.set_hatch(hatch_map[method])
        bar.set_linewidth(2.5)

    ax.set_xticks(x)
    ax.set_xticklabels([])  # Hide x-tick labels as they're handled elsewhere

    return bars_sufficient, bars_pressure

def add_exclusive_line(ax, exclusive_val):
    """Add exclusive baseline as horizontal line."""
    if exclusive_val is not None:
        ax.axhline(y=exclusive_val, color=EXCLUSIVE_LINE_COLOR,
                  linestyle=EXCLUSIVE_LINE_STYLE, linewidth=EXCLUSIVE_LINE_WIDTH)

def reduce_vllm_spacing(fig, vllm_axes, diff_x0):
    """Reduce horizontal spacing between vLLM subplots."""
    first_pos = vllm_axes[0].get_position()
    first_x0 = first_pos.x0
    subplot_width = first_pos.x1 - first_pos.x0
    available_width = diff_x0 - first_x0
    gap_width = max((available_width - 4 * subplot_width) / 3 * VLLM_SPACING_REDUCTION, MIN_SUBPLOT_GAP)

    current_x = first_x0
    for ax in vllm_axes:
        pos = ax.get_position()
        ax.set_position([current_x, pos.y0, subplot_width, pos.y1 - pos.y0])
        current_x += subplot_width + gap_width

def main():
    parser = argparse.ArgumentParser(description="Script for drawing performance bar chart with memory pressure overlay")
    parser.add_argument("--input", type=str, required=True, help="Path to memory sufficient input data")
    parser.add_argument("--input-pressure", type=str, required=True,
                       help="Path to memory pressure data (for overlapped bars)")
    parser.add_argument("--output", type=str, required=True, help="Path to output fig")
    args = parser.parse_args()

    # Load data
    results, applications = load_data(args.input)
    print(results)

    # Load memory pressure data
    results_pressure, _ = load_data(args.input_pressure)
    print("Memory pressure data:", results_pressure)

    # Setup colors and patterns
    color_map = get_color_map()
    bar_methods = [m for m in METHODS_ORDER if m != 'exclusive']
    hatch_map = {method: HATCH_PATTERNS[i % len(HATCH_PATTERNS)] for i, method in enumerate(bar_methods)}

    # Setup matplotlib with Type 1 fonts for OSDI submission
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
        # Type 1 font configuration for OSDI submission
        'pdf.fonttype': 42,  # Embed fonts as Type 1
        'ps.fonttype': 42,   # Embed fonts as Type 1
        'font.family': 'sans-serif',  # Use sans-serif fonts (Type 1 compatible)
    })

    # Create figure
    fig = plt.figure(figsize=FIGURE_SIZE)

    # Application labels
    def format_app_label(app):
        app_lower = app.lower()
        if app_lower == 'vllm':
            return 'vLLM'
        elif app_lower == 'diffusion':
            return 'Diffusion'
        elif app_lower == 'llama_factory':
            return 'LlamaFactory'
        else:
            return app.capitalize()

    app0_label = format_app_label(applications[0])
    app1_label = format_app_label(applications[1])

    # Plot vLLM metrics (subplots 1-4)
    for i, (src_key, display_label, scale, use_log) in enumerate(VLLM_METRICS, start=1):
        ax = fig.add_subplot(1, 5, i)
        actual_src_key = src_key  # Use the actual key directly

        values_sufficient = [results['vllm'][method][actual_src_key] * scale for method in bar_methods]

        # Get memory pressure values, fallback to sufficient values if missing
        if results_pressure and 'vllm' in results_pressure:
            values_pressure = [results_pressure['vllm'][method][actual_src_key] * scale
                              if method in results_pressure['vllm'] and actual_src_key in results_pressure['vllm'][method]
                              else values_sufficient[j]  # Fallback to sufficient value if pressure data missing
                              for j, method in enumerate(bar_methods)]
        else:
            # If no pressure data, use sufficient values for both
            values_pressure = values_sufficient.copy()

        # Create grouped bars side-by-side
        bars_sufficient, bars_pressure = create_grouped_bar_plot(
            ax, values_sufficient, values_pressure, bar_methods, color_map, hatch_map)

        # Add exclusive line
        exclusive_val = (results['vllm']['exclusive'][actual_src_key] * scale
                        if 'exclusive' in results['vllm'] else None)
        add_exclusive_line(ax, exclusive_val)

        # Configure axis
        # Collect all values including memory pressure for axis limits
        all_values = values_sufficient + values_pressure + ([exclusive_val] if exclusive_val is not None else [])

        if use_log:
            ax.set_yscale('log')
            # Set y-axis to start near 0 for TTFT metrics
            min_val = min([v for v in all_values if v > 0])
            ax.set_ylim(bottom=min_val * 0.1)  # Start at 10% of minimum value (near 0)
        else:
            ax.set_ylim(bottom=0)  # Linear scale starts at 0
            # Cap P99 ITL y-axis at 650 only if any values exceed it
            if src_key == 'P99 ITL (ms)':
                max_val = max(all_values)
                if max_val > 650:
                    ax.set_ylim(top=650)
                    # Add text labels for bars exceeding 650
                    x_pos = np.arange(len(bar_methods))
                    width = 0.35
                    for j, (value_s, value_p) in enumerate(zip(values_sufficient, values_pressure)):
                        if value_s > 650:
                            ax.text(x_pos[j] - width/2, 650,
                                   f'{value_s:.0f}', ha='center', va='bottom',
                                   fontsize=FONT_SIZE - 1, color='tab:red')
                        if value_p > 650:
                            ax.text(x_pos[j] + width/2, 650,
                                   f'{value_p:.0f}', ha='center', va='bottom',
                                   fontsize=FONT_SIZE - 1, color='tab:red')
        ax.set_ylabel(display_label)

    # Plot throughput metric (subplot 5) - separate group with spacing
    ax = fig.add_subplot(1, 5, 5)
    latencies_sufficient = [results[applications[1]][method]['latency'] for method in bar_methods]
    exclusive_latency = (results[applications[1]]['exclusive']['latency']
                        if 'exclusive' in results[applications[1]] else None)

    if exclusive_latency is not None:
        norm_throughput_sufficient = [exclusive_latency / L for L in latencies_sufficient]
    else:
        norm_throughput_sufficient = [1 for _ in latencies_sufficient]

    # Get memory pressure throughput values
    if results_pressure and applications[1] in results_pressure:
        latencies_pressure = [results_pressure[applications[1]][method]['latency']
                             if method in results_pressure[applications[1]] and 'latency' in results_pressure[applications[1]][method]
                             else latencies_sufficient[j]  # Fallback to sufficient latency if missing
                             for j, method in enumerate(bar_methods)]
        exclusive_latency_pressure = (results_pressure[applications[1]]['exclusive']['latency']
                                      if 'exclusive' in results_pressure[applications[1]] and 'latency' in results_pressure[applications[1]]['exclusive']
                                      else exclusive_latency)

        if exclusive_latency_pressure is not None:
            norm_throughput_pressure = [exclusive_latency_pressure / L for L in latencies_pressure]
        else:
            norm_throughput_pressure = [1 for _ in latencies_pressure]
    else:
        # If no pressure data, use sufficient values for both
        norm_throughput_pressure = norm_throughput_sufficient.copy()

    # Create grouped bars side-by-side
    bars_sufficient, bars_pressure = create_grouped_bar_plot(
        ax, norm_throughput_sufficient, norm_throughput_pressure, bar_methods, color_map, hatch_map)

    ax.set_ylabel("Norm. Tput. (x)")

    # Adjust layout and spacing
    fig.tight_layout(rect=[0, 0.20, 1, 1])

    # Reduce horizontal spacing between vLLM subplots
    vllm_axes = [fig.axes[i] for i in range(4)]
    ax_diff = fig.axes[4]
    reduce_vllm_spacing(fig, vllm_axes, ax_diff.get_position().x0)

    # Add visual separator between groups
    ax_vllm_first, ax_vllm_last = fig.axes[0], fig.axes[3]
    vllm_last_pos = ax_vllm_last.get_position()
    separator_x = (vllm_last_pos.x1 + ax_diff.get_position().x0) / 2 - 0.025
    fig.add_artist(Line2D([separator_x, separator_x],
                         [vllm_last_pos.y0 - 0.15, vllm_last_pos.y1 + 0.03],
                         color='gray', linewidth=2, linestyle='--', alpha=0.6,
                         transform=fig.transFigure, zorder=0))

    # Add subfigure captions
    vllm_center_x = (ax_vllm_first.get_position().x0 + ax_vllm_last.get_position().x1) / 2
    diff_center_x = (ax_diff.get_position().x0 + ax_diff.get_position().x1) / 2 - (0.03 if app1_label == 'LlamaFactory' else 0.025)
    fig.text(vllm_center_x, 0.15, f"{app0_label} (↓ is better)",
             fontsize=FONT_SIZE, ha='center', va='bottom', transform=fig.transFigure)
    fig.text(diff_center_x, 0.15, f"{app1_label} (↑ is better)",
             fontsize=FONT_SIZE, ha='center', va='bottom', transform=fig.transFigure)

    # Add legend - clearly show memory sufficient and memory pressure
    # Show methods (representing both sufficient and pressure with same pattern, different alpha in plot)
    legend_handles = [
        mpatches.Patch(facecolor=color_map[method] if method == 'GVM' else 'white',
                      edgecolor='black' if method == 'GVM' else color_map[method],
                      hatch=hatch_map[method], linewidth=2.5, label=method)
        for method in bar_methods
    ]
    # Add indicators for the two conditions
    legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='black',
                                         hatch=None, linewidth=2.5,
                                         alpha=0.7, label='Mem. Sufficient'))
    legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='black',
                                         hatch=None, linewidth=2.5,
                                         alpha=0.6, label='Mem. Pressure'))
    legend_handles.append(Line2D([], [], color=EXCLUSIVE_LINE_COLOR,
                                linestyle=EXCLUSIVE_LINE_STYLE,
                                linewidth=EXCLUSIVE_LINE_WIDTH, label='Exclusive'))

    if SHOW_LEGEND:
        fig.legend(handles=legend_handles, loc=(0.23, 0.88),
                   ncol=len(legend_handles), frameon=False)

    # Save plot
    filename = f"{'+'.join(applications)}_{args.output}.pdf"
    fig.savefig(filename, bbox_inches='tight')

if __name__ == "__main__":
    main()
