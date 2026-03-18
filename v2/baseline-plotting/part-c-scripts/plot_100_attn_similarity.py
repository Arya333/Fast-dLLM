# This script plots the attention-weight similarity heatmap (across 100 samples) from the baseline trace logs.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOG_DIR = Path("baseline_logs_100_attn_similarity")
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

    for record in data["attn_weight_records"]:
        max_layer = max(max_layer, record["layer_idx"])
        max_step = max(max_step, record["step_idx"])

num_layers = max_layer + 1
num_steps = max_step + 1

sample_matrices = []

for data in all_data:
    layer_step_sum = np.zeros((num_layers, num_steps), dtype=np.float64)
    layer_step_count = np.zeros((num_layers, num_steps), dtype=np.float64)

    for record in data["attn_weight_records"]:
        layer_idx = record["layer_idx"]
        step_idx = record["step_idx"]
        sim = record["cosine_similarity"]

        layer_step_sum[layer_idx, step_idx] += sim
        layer_step_count[layer_idx, step_idx] += 1.0

    sample_matrix = np.full((num_layers, num_steps), np.nan, dtype=np.float64)

    for layer_idx in range(num_layers):
        for step_idx in range(num_steps):
            if layer_step_count[layer_idx, step_idx] > 0:
                sample_matrix[layer_idx, step_idx] = (
                    layer_step_sum[layer_idx, step_idx]
                    / layer_step_count[layer_idx, step_idx]
                )
    sample_matrices.append(sample_matrix[:, 1:])

sample_matrices = np.stack(sample_matrices, axis=0)

mean_heatmap = np.nanmean(sample_matrices, axis=0)

plt.figure(figsize=(10, 6))
plt.imshow(mean_heatmap, aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="Mean cosine similarity")
plt.xlabel("Denoising step")
plt.ylabel("Layer")
plt.title("100-sample attention similarity mean")
plt.tight_layout()
plt.savefig(OUT_DIR / "attn_similarity_100_mean.png", dpi=200)
plt.close()


print("Saved:")
print(OUT_DIR / "attn_similarity_100_mean.png")
print("Num samples:", len(json_paths))
