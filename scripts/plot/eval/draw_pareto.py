import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description="Script for drawing pareto fig")

parser.add_argument("--input", type=str, required=True, help="Path to input data")
parser.add_argument("--output", type=str, required=True, help="Path to output fig")
parser.add_argument("--slo", type=float, required=True, help="SLO for counting SLO attainment")

def calculate_metric(application, filepath, slo):
    if application == "vllm":
        total_tokens = 0
        slo_satisfied_tokens = 0
        fileobj = open(filepath)
        data = json.load(fileobj)
        fileobj.close()

        average_output_tokens = data["total_output_tokens"] // data["completed"]

        for itl in data["itls"]:
            if len(itl) == 0:
                total_tokens += average_output_tokens
                continue

            for latency in itl:
                if latency <= slo:
                    slo_satisfied_tokens += 1
                total_tokens += 1

        return slo_satisfied_tokens / np.max([total_tokens, 1])
    elif application == "diffusion":
        return 1 / np.mean(np.loadtxt(filepath))
    else:
        raise NotImplementedError("Application: {} is not supported".format(application))


if __name__ == "__main__":
    args = parser.parse_args()

    applications = os.path.basename(os.path.normpath(args.input)).split("+")
    results = { application : {} for application in applications }

    # First pass: collect all data including exclusive baselines
    for file in os.listdir(args.input):
        application = file.split("-")[0]
        method = "-".join(file.split("-")[1:-2])
        # Filter out "-28G" data points
        if "-28G" in method:
            continue
        results[application][method] = calculate_metric(application, os.path.join(args.input, file), args.slo)

    # Get diffusion_exclusive baseline for normalization
    diffusion_exclusive_baseline = results["diffusion"].get("diffusion-exclusive", None)

    # Extract common keys (methods) - only use keys that exist in both datasets
    vllm_keys = set(results["vllm"].keys())
    diffusion_keys = set(results["diffusion"].keys())
    common_keys = vllm_keys.intersection(diffusion_keys)
    # Filter out exclusive data points from plotting
    common_keys = [k for k in common_keys if k not in ["diffusion-exclusive", "vllm-exclusive"]]
    keys = sorted(common_keys)

    # Define color scheme and markers matching bar chart
    palette = plt.get_cmap('tab10').colors
    methods_color_map = {
        'TGS': palette[0],
        'GPreempt': palette[1],
        'xsched': palette[2],
        'GVM': 'tab:purple'
    }

    # Define markers for different systems
    methods_marker_map = {
        'TGS': 'o',
        'GPreempt': 's',
        'xsched': '^',
        'GVM': 'D'
    }

    # Categorize methods by system
    def get_system(method):
        if method.startswith('GVM'):
            return 'GVM'
        elif method == 'TGS':
            return 'TGS'
        elif method == 'GPreempt':
            return 'GPreempt'
        elif method == 'xsched':
            return 'xsched'
        else:
            return 'other'

    # Group methods by system
    system_groups = {}
    for method in keys:
        system = get_system(method)
        if system not in system_groups:
            system_groups[system] = []
        system_groups[system].append(method)

    x = [results["vllm"][k] for k in keys]         # vllm performance (x-axis)
    # Normalize diffusion throughput to diffusion_exclusive baseline
    if diffusion_exclusive_baseline is not None:
        y = [results["diffusion"][k] / diffusion_exclusive_baseline for k in keys]    # normalized diffusion performance (y-axis)
    else:
        y = [results["diffusion"][k] for k in keys]    # diffusion performance (y-axis)

    # Set font sizes
    FONT_SIZE = 15
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
    })

    plt.figure(figsize=(4.5, 4.5))

    # Plot points grouped by system with different colors and markers
    legend_handles = []
    # Define legend order to match bar chart (excluding exclusive)
    legend_order = ['TGS', 'GPreempt', 'xsched', 'GVM']

    for system in legend_order:
        if system in system_groups and system in methods_color_map:
            methods = system_groups[system]
            system_x = [results["vllm"][method] for method in methods]
            system_y = [results["diffusion"][method] / diffusion_exclusive_baseline for method in methods] if diffusion_exclusive_baseline is not None else [results["diffusion"][method] for method in methods]

            # Convert x values to percentage scale (multiply by 100)
            system_x_percent = [x * 100 for x in system_x]

            # Plot scatter points
            plt.scatter(system_x_percent, system_y,
                       color=methods_color_map[system],
                       marker=methods_marker_map[system],
                       s=80, alpha=0.8, edgecolors='black', linewidth=1)

            # Connect GVM dots with a line
            if system == 'GVM' and len(system_x_percent) > 1:
                # Sort points by x-coordinate to create a proper line
                sorted_indices = sorted(range(len(system_x_percent)), key=lambda i: system_x_percent[i])
                sorted_x = [system_x_percent[i] for i in sorted_indices]
                sorted_y = [system_y[i] for i in sorted_indices]
                plt.plot(sorted_x, sorted_y, color=methods_color_map[system],
                        linestyle='-', linewidth=2, alpha=0.6)

            # Add to legend
            legend_handles.append(Line2D([], [], color=methods_color_map[system],
                                       marker=methods_marker_map[system],
                                       linestyle='None', markersize=8,
                                       label=system if system != 'xsched' else 'XSched'))

    # # Label each point (simplified labels for GVM)
    # for i, k in enumerate(keys):
    #     display_label = k
    #     if k.startswith('GVM'):
    #         # Extract the configuration part (e.g., "2-10" from "GVM-2-10")
    #         parts = k.split('-')
    #         if len(parts) >= 3:
    #             display_label = f"GVM-{parts[1]}-{parts[2]}"
    #     elif k == "xsched":
    #         display_label = "XSched"

    #     plt.text(x[i] + 0.002, y[i] + 0.002, display_label, fontsize=12)

    plt.xlabel("vLLM SLO attainment (%)", fontsize=16)
    plt.ylabel("Diffusion throughput (x)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    # Set axes ranges
    plt.xlim(0, 100)  # Set x-axis range to (0,100) for percentage
    plt.ylim(0, 1)  # Set y-axis range to (0,1)


    # Add legend
    plt.legend(handles=legend_handles, loc=(0.48, .65), frameon=False,fontsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(args.output, "{}_pareto.pdf".format("+".join(applications))))
