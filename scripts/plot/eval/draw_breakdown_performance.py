import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as patches

parser = argparse.ArgumentParser(description="Script for drawing breakdown performance chart")

parser.add_argument("--input", type=str, required=True, help="Path to input data")
parser.add_argument("--output", type=str, required=True, help="Path to output fig")

def calculate_metric(application, filepath):
	if application == "vllm":
		total_tokens = 0
		fileobj = open(filepath)
		data = json.load(fileobj)
		fileobj.close()

		return {
			"Median TTFT (ms)" : data["median_ttft_ms"],
			"p99 TTFT (ms)" : data["p99_ttft_ms"],
			"Median ITL (ms)" : data["median_itl_ms"],
			"p99 ITL (ms)" : data["p99_itl_ms"],
		}
	elif application == "diffusion":
		return { "latency" : np.mean(np.loadtxt(filepath)) }
	elif application == "llama_factory":
		return { "latency" : np.loadtxt(filepath) }
	else:
		raise NotImplementedError("Application: {} is not supported".format(application))


if __name__ == "__main__":
	args = parser.parse_args()

	applications = os.path.basename(os.path.normpath(args.input)).split("+")
	results = { application : {} for application in applications }

	for file in os.listdir(args.input):
		application = file.split("-")[0]
		# Parse method name by removing timestamp (last part after last dash)
		parts = file.split("-")
		# Find the timestamp part (starts with year 2025)
		timestamp_idx = -1
		for i, part in enumerate(parts):
			if part.startswith("2025"):
				timestamp_idx = i
				break

		if timestamp_idx > 0:
			method = "-".join(parts[1:timestamp_idx])
		else:
			method = "-".join(parts[1:-1])  # fallback to original logic

		results[application][method] = calculate_metric(application, os.path.join(args.input, file))

	print(results)

	# Define breakdown configurations in order of improvement
	breakdown_configs = ['UVM', 'GVM-Part-1-Mem-Iso', 'GVM-Part-2-Com-Iso', 'GVM-Part-3-Swap-Opt']
	config_labels = ['UVM', 'Mem-Iso', 'Com-Iso', 'Swap-Opt']

	# Color scheme for breakdown configurations
	palette = ['tab:red', 'tab:orange', 'tab:green', 'tab:purple']
	config_color_map = { config: palette[i] for i, config in enumerate(breakdown_configs) }

	# Define mapping for display: (source_key_in_results, display_label, scale_factor)
	vllm_plot_map = [
		('Median TTFT (ms)', 'Median TTFT (s)', 0.001),
		('p99 TTFT (ms)', 'P99 TTFT (s)', 0.001),
		('Median ITL (ms)', 'Median ITL (ms)', 1.0),
		('p99 ITL (ms)', 'P99 ITL (ms)', 1.0),
	]

	# Nicely formatted application labels
	app0_label = 'vLLM' if applications[0].lower() == 'vllm' else applications[0].capitalize()
	app1_label = 'Diffusion' if applications[1].lower() == 'diffusion' else applications[1].capitalize()

	# Global font sizing
	FONT_SIZE = 15
	plt.rcParams.update({
		'font.size': FONT_SIZE,
		'axes.labelsize': FONT_SIZE,
		'xtick.labelsize': FONT_SIZE,
		'ytick.labelsize': FONT_SIZE,
		'legend.fontsize': FONT_SIZE,
	})

	# Create a figure with 5 subplots (1 row, 5 columns)
	plt.figure(figsize=(16, 4))

	# 1–4: vLLM metrics (TTFT in seconds, ITL in ms)
	for i, (src_key, display_label, scale) in enumerate(vllm_plot_map, start=1):
		plt.subplot(1, 5, i)

		# Get values for each configuration
		values = []
		colors = []
		for config in breakdown_configs:
			if config in results['vllm']:
				values.append(results['vllm'][config][src_key] * scale)
				colors.append(config_color_map[config])
			else:
				values.append(0)
				colors.append('gray')

		bars = plt.bar(config_labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

		# Add improvement arrows between consecutive configurations
		for j in range(len(values) - 1):
			if values[j] > 0 and values[j+1] > 0:
				# Calculate improvement percentage
				improvement = (values[j] - values[j+1]) / values[j] * 100
				if improvement > 0:  # Only show arrows for improvements
					# Arrow from current bar to next bar
					arrow = FancyArrowPatch(
						(j, values[j] * 1.1), (j+1, values[j+1] * 1.1),
						arrowstyle='->', mutation_scale=20, color='black', linewidth=2
					)
					plt.gca().add_patch(arrow)

					# Add improvement percentage text
					mid_x = (j + j+1) / 2
					mid_y = max(values[j], values[j+1]) * 1.2
					plt.text(mid_x, mid_y, f'{improvement:.1f}%',
							ha='center', va='bottom', fontsize=10, fontweight='bold')

		# Log scale for TTFT metrics (only if all values are positive)
		if 'TTFT' in src_key and all(v > 0 for v in values):
			plt.yscale('log')

		plt.ylabel("{} {}".format(app0_label, display_label))
		plt.xticks(rotation=45)
		plt.grid(True, alpha=0.3)

	# 5: Diffusion normalized throughput vs UVM baseline
	plt.subplot(1, 5, 5)
	latencies = []
	colors = []
	for config in breakdown_configs:
		if config in results[applications[1]]:
			latencies.append(results[applications[1]][config]['latency'])
			colors.append(config_color_map[config])
		else:
			latencies.append(1)
			colors.append('gray')

	# Normalize throughput to UVM baseline: (1/L) / (1/L_UVM) = L_UVM / L
	uvm_latency = results[applications[1]]['UVM']['latency'] if 'UVM' in results[applications[1]] else None
	if uvm_latency is not None:
		norm_throughput = [uvm_latency / L for L in latencies]
	else:
		norm_throughput = [1 for _ in latencies]

	bars = plt.bar(config_labels, norm_throughput, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

	# Add improvement arrows for diffusion
	for j in range(len(norm_throughput) - 1):
		if norm_throughput[j] > 0 and norm_throughput[j+1] > 0:
			# Calculate improvement percentage
			improvement = (norm_throughput[j+1] - norm_throughput[j]) / norm_throughput[j] * 100
			if improvement > 0:  # Only show arrows for improvements
				# Arrow from current bar to next bar
				arrow = FancyArrowPatch(
					(j, norm_throughput[j] * 1.1), (j+1, norm_throughput[j+1] * 1.1),
					arrowstyle='->', mutation_scale=20, color='black', linewidth=2
				)
				plt.gca().add_patch(arrow)

				# Add improvement percentage text
				mid_x = (j + j+1) / 2
				mid_y = max(norm_throughput[j], norm_throughput[j+1]) * 1.2
				plt.text(mid_x, mid_y, f'{improvement:.1f}%',
						ha='center', va='bottom', fontsize=10, fontweight='bold')

	plt.ylabel("{} Norm. Tput.".format(app1_label))
	plt.xticks(rotation=45)
	plt.grid(True, alpha=0.3)

	# Legend (top center) with configuration colors
	legend_handles = [
		mpatches.Patch(facecolor=config_color_map[config], alpha=0.7, label=config_labels[i])
		for i, config in enumerate(breakdown_configs)
	]
	plt.tight_layout(rect=[0, 0.15, 1, 1])
	plt.figlegend(handles=legend_handles, loc=(0.25, .9), ncol=len(legend_handles), frameon=False)

	plt.savefig(os.path.join(args.output, "{}_breakdown_performance.pdf".format("+".join(applications))), bbox_inches='tight')
