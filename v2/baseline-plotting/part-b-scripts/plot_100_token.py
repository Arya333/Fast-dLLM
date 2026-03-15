import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

sample_mean_matrices = []
sample_var_matrices = []

for data in all_data:
    # For each sample:
    # mean_matrix[layer, step] = average token cosine similarity
    # var_matrix[layer, step] = variance across token cosine similarities
    layer_step_mean_sum = np.zeros((num_layers, num_steps), dtype=np.float64)
    layer_step_var_sum = np.zeros((num_layers, num_steps), dtype=np.float64)
    layer_step_count = np.zeros((num_layers, num_steps), dtype=np.float64)

    for record in data["records"]:
        layer_idx = record["layer_idx"]
        step_idx = record["step_idx"]
        token_cos = np.array(record["token_cosine"], dtype=np.float64)

        token_mean = float(np.mean(token_cos))
        token_var = float(np.var(token_cos))

        layer_step_mean_sum[layer_idx, step_idx] += token_mean
        layer_step_var_sum[layer_idx, step_idx] += token_var
        layer_step_count[layer_idx, step_idx] += 1.0

    mean_matrix = np.full((num_layers, num_steps), np.nan, dtype=np.float64)
    var_matrix = np.full((num_layers, num_steps), np.nan, dtype=np.float64)

    for layer_idx in range(num_layers):
        for step_idx in range(num_steps):
            if layer_step_count[layer_idx, step_idx] > 0:
                mean_matrix[layer_idx, step_idx] = (
                    layer_step_mean_sum[layer_idx, step_idx]
                    / layer_step_count[layer_idx, step_idx]
                )
                var_matrix[layer_idx, step_idx] = (
                    layer_step_var_sum[layer_idx, step_idx]
                    / layer_step_count[layer_idx, step_idx]
                )

    # Step 0 has no t-1 comparison, so drop it
    sample_mean_matrices.append(mean_matrix[:, 1:])
    sample_var_matrices.append(var_matrix[:, 1:])

sample_mean_matrices = np.stack(sample_mean_matrices, axis=0)
sample_var_matrices = np.stack(sample_var_matrices, axis=0)

mean_heatmap = np.nanmean(sample_mean_matrices, axis=0)
var_heatmap = np.nanmean(sample_var_matrices, axis=0)


plt.figure(figsize=(10, 6))
plt.imshow(mean_heatmap, aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="Mean per-token similarity")
plt.xlabel("Denoising step")
plt.ylabel("Layer")
plt.title("100-sample token similarity mean (for token skipping motivation)")
plt.tight_layout()
plt.savefig(OUT_DIR / "token_100_mean.png", dpi=200)
plt.close()


plt.figure(figsize=(10, 6))
plt.imshow(var_heatmap, aspect="auto", origin="lower", cmap="magma")
plt.colorbar(label="Mean per-token variance")
plt.xlabel("Denoising step")
plt.ylabel("Layer")
plt.title("100-sample token similarity variance (for token skipping motivation)")
plt.tight_layout()
plt.savefig(OUT_DIR / "token_100_variance.png", dpi=200)
plt.close()

print("Saved:")
print(OUT_DIR / "token_100_mean.png")
print(OUT_DIR / "token_100_variance.png")
print("Num samples:", len(json_paths))
