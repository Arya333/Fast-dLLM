# This script plots the normalized layer-level hidden-state similarity heatmap (across 100 samples) from the baseline trace logs.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def normalize_each_step(sample_matrix):
    # Normalize each denoising step column separately across layers
    normalized = sample_matrix.copy()

    for step_idx in range(normalized.shape[1]):
        col = normalized[:, step_idx]
        valid = ~np.isnan(col)

        if not np.any(valid):
            continue

        col_min = np.nanmin(col)
        col_max = np.nanmax(col)

        if col_max > col_min:
            normalized[valid, step_idx] = (col[valid] - col_min) / (col_max - col_min)
        else:
            # If every layer has the same value at this step, set the column to 0
            normalized[valid, step_idx] = 0.0

    return normalized

LOG_DIR = Path("baseline_logs_100")
OUT_DIR = Path("baseline-plotting")
OUT_DIR.mkdir(exist_ok=True)

json_paths = sorted(LOG_DIR.glob("sample_*.json"))
if len(json_paths) == 0:
    raise ValueError("No sample json files found.")

all_data = []
max_layer = 0
max_step = 0

for path in json_paths:
    with open(path, "r") as f:
        data = json.load(f)
    all_data.append(data)

    for record in data["records"]:
        max_layer = max(max_layer, record["layer_idx"])
        max_step = max(max_step, record["step_idx"])

num_layers = max_layer + 1
num_steps = max_step + 1

sample_matrices = []

for data in all_data:
    layer_step_sum = np.zeros((num_layers, num_steps), dtype=np.float64)
    layer_step_count = np.zeros((num_layers, num_steps), dtype=np.float64)

    for record in data["records"]:
        layer_idx = record["layer_idx"]
        step_idx = record["step_idx"]

        layer_step_sum[layer_idx, step_idx] += record["mean_cosine"]
        layer_step_count[layer_idx, step_idx] += 1.0

    sample_matrix = np.full((num_layers, num_steps), np.nan, dtype=np.float64)

    for layer_idx in range(num_layers):
        for step_idx in range(num_steps):
            if layer_step_count[layer_idx, step_idx] > 0:
                sample_matrix[layer_idx, step_idx] = (
                    layer_step_sum[layer_idx, step_idx] / layer_step_count[layer_idx, step_idx]
                )

    # Step 0 has no t-1 comparison, so drop it
    sample_matrix = sample_matrix[:, 1:]

    # Normalize each step separately so different denoising steps are comparable
    sample_matrix = normalize_each_step(sample_matrix)

    sample_matrices.append(sample_matrix)

sample_matrices = np.stack(sample_matrices, axis=0)

mean_heatmap = np.nanmean(sample_matrices, axis=0)
var_heatmap = np.nanvar(sample_matrices, axis=0)


plt.figure(figsize=(10, 6))
plt.imshow(mean_heatmap, aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="Normalized mean cosine similarity")
plt.xlabel("Denoising step")
plt.ylabel("Layer")
plt.title("100-sample hidden-state similarity mean (with step-normalization; for layer skipping motivation)")
plt.tight_layout()
plt.savefig(OUT_DIR / "layer_100_mean_normalized.png", dpi=200)
plt.close()


plt.figure(figsize=(10, 6))
plt.imshow(var_heatmap, aspect="auto", origin="lower", cmap="magma")
plt.colorbar(label="Variance of normalized similarity")
plt.xlabel("Denoising step")
plt.ylabel("Layer")
plt.title("100-sample hidden-state similarity variance (with step-normalization; for layer skipping motivation)")
plt.tight_layout()
plt.savefig(OUT_DIR / "layer_100_variance_normalized.png", dpi=200)
plt.close()

print("Saved:")
print(OUT_DIR / "layer_100_mean.png")
print(OUT_DIR / "layer_100_variance.png")
print("Num samples:", len(json_paths))
