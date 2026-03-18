# This script plots the attention-weight value-range distribution (across 100 samples) from the baseline trace logs.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOG_DIR = Path("baseline_logs_100_attn_range")
OUT_DIR = Path("baseline-plotting")
OUT_DIR.mkdir(parents=True, exist_ok=True)

json_paths = sorted(LOG_DIR.glob("sample_*.json"))
if len(json_paths) == 0:
    raise ValueError("No sample json files found.")

num_bins = 40

# Aggregate over all samples, steps, layers, and tokens
total_bin_counts = np.zeros(num_bins, dtype=np.float64)

global_total = 0.0
global_in_range = 0.0
global_below = 0.0
global_above = 0.0

for path in json_paths:
    with open(path, "r") as f:
        data = json.load(f)

    for record in data["attn_range_records"]:
        total_bin_counts += np.array(record["bin_counts"], dtype=np.float64)

        global_total += float(record["total_count"])
        global_in_range += float(record["num_in_range"])
        global_below += float(record["num_below_range"])
        global_above += float(record["num_above_range"])

if global_total > 0:
    heatmap = total_bin_counts.reshape(1, num_bins)
else:
    heatmap = np.zeros((1, num_bins), dtype=np.float64)

in_range_pct = 100.0 * global_in_range / global_total if global_total > 0 else 0.0
below_pct = 100.0 * global_below / global_total if global_total > 0 else 0.0
above_pct = 100.0 * global_above / global_total if global_total > 0 else 0.0

plt.figure(figsize=(14, 3.2))
plt.imshow(heatmap, aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="Occurrence count")
plt.xlabel("Attention weight range")
plt.ylabel("Occurrence")
plt.title("Attention score range heatmap (100 samples)")

tick_positions = [-0.5, 9.5, 19.5, 29.5, 39.5]
tick_labels = ["1e-4", "1e-3", "1e-2", "1e-1", "1"]
plt.xticks(tick_positions, tick_labels)

plt.yticks([])
plt.ylabel("Occurrence")

for x in [9.5, 19.5, 29.5]:
    plt.axvline(x=x, color="white", linestyle="--", linewidth=0.8, alpha=0.6)

summary_text = (
    f"In plotted range [1e-4, 1]: {int(global_in_range)} ({in_range_pct:.2f}%)   "
    f"< 1e-4: {int(global_below)} ({below_pct:.2f}%)   "
    f"> 1: {int(global_above)} ({above_pct:.2f}%)"
)
plt.figtext(0.5, 0.01, summary_text, ha="center", fontsize=10)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(OUT_DIR / "attn_range_100_heatmap.png", dpi=200)
plt.close()

print("Saved:")
print(OUT_DIR / "attn_range_100_heatmap.png")
print("Num samples:", len(json_paths))
print(summary_text)
