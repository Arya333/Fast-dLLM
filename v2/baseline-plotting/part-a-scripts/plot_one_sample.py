# import json
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np


# JSON_PATH = Path("baseline_logs/sample_0000.json")
# OUT_DIR = Path("plotting")
# OUT_DIR.mkdir(exist_ok=True)


# with open(JSON_PATH, "r") as f:
#     data = json.load(f)

# records = data["records"]

# max_layer = max(r["layer_idx"] for r in records)
# max_step = max(r["step_idx"] for r in records)
# num_tokens = len(records[0]["token_cosine"])

# num_layers = max_layer + 1
# num_steps = max_step + 1

# # Average over blocks for the layer-vs-step plot
# layer_step_sum = np.zeros((num_layers, num_steps), dtype=np.float64)
# layer_step_count = np.zeros((num_layers, num_steps), dtype=np.float64)

# # Average over all blocks and steps for the layer-vs-token plot
# layer_token_sum = np.zeros((num_layers, num_tokens), dtype=np.float64)
# layer_token_count = np.zeros((num_layers, num_tokens), dtype=np.float64)

# for record in records:
#     layer_idx = record["layer_idx"]
#     step_idx = record["step_idx"]
#     token_cos = np.array(record["token_cosine"], dtype=np.float64)

#     layer_step_sum[layer_idx, step_idx] += record["mean_cosine"]
#     layer_step_count[layer_idx, step_idx] += 1

#     layer_token_sum[layer_idx] += token_cos
#     layer_token_count[layer_idx] += 1

# layer_step_mean = layer_step_sum / np.maximum(layer_step_count, 1.0)
# layer_token_mean = layer_token_sum / np.maximum(layer_token_count, 1.0)

# # Do not care about step 0
# layer_step_mean = layer_step_mean[:, 1:]


# plt.figure(figsize=(10, 6))
# plt.imshow(layer_step_mean, aspect="auto", origin="lower", cmap="viridis")
# plt.colorbar(label="Mean cosine similarity")
# plt.xlabel("Denoising step")
# plt.ylabel("Layer")
# plt.title("One-sample layer similarity heatmap")
# plt.tight_layout()
# plt.savefig(OUT_DIR / "one_sample_layer_heatmap.png", dpi=200)
# plt.close()


# plt.figure(figsize=(10, 6))
# plt.imshow(layer_token_mean, aspect="auto", origin="lower", cmap="viridis")
# plt.colorbar(label="Mean cosine similarity")
# plt.xlabel("Token position in block")
# plt.ylabel("Layer")
# plt.title("One-sample token similarity heatmap")
# plt.tight_layout()
# plt.savefig(OUT_DIR / "one_sample_token_heatmap.png", dpi=200)
# plt.close()

# print("Saved:")
# print(OUT_DIR / "one_sample_layer_heatmap.png")
# print(OUT_DIR / "one_sample_token_heatmap.png")

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


JSON_PATH = Path("baseline_logs/sample_0000.json")
OUT_DIR = Path("baseline-plotting")
OUT_DIR.mkdir(exist_ok=True)


with open(JSON_PATH, "r") as f:
    data = json.load(f)

records = data["records"]
prompt_len = data["prompt_len"]

max_layer = max(r["layer_idx"] for r in records)
max_step = max(r["step_idx"] for r in records)
max_block_idx = max(r["block_idx"] for r in records)

block_size = len(records[0]["token_cosine"])
num_layers = max_layer + 1
num_steps = max_step + 1

first_block_idx = prompt_len // block_size
prompt_offset = prompt_len % block_size

# -----------------------------
# Layer-level heatmap
# x-axis: denoising step
# y-axis: layer
# value: mean cosine similarity, averaged over blocks
# -----------------------------
layer_step_sum = np.zeros((num_layers, num_steps), dtype=np.float64)
layer_step_count = np.zeros((num_layers, num_steps), dtype=np.float64)

for record in records:
    layer_idx = record["layer_idx"]
    step_idx = record["step_idx"]

    layer_step_sum[layer_idx, step_idx] += record["mean_cosine"]
    layer_step_count[layer_idx, step_idx] += 1.0

layer_step_mean = layer_step_sum / np.maximum(layer_step_count, 1.0)

# Step 0 has no t-1 similarity, so do not plot it
layer_step_mean = layer_step_mean[:, 1:]


plt.figure(figsize=(10, 6))
plt.imshow(layer_step_mean, aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="Mean cosine similarity")
plt.xlabel("Denoising step")
plt.ylabel("Layer")
plt.title("One-sample layer similarity heatmap")
plt.tight_layout()
plt.savefig(OUT_DIR / "one_sample_layer_heatmap.png", dpi=200)
plt.close()


# -----------------------------
# Token-level heatmap
# x-axis: generated token position across all generated blocks
# y-axis: layer
# value: mean cosine similarity, averaged over denoising steps
# -----------------------------
total_generated_positions = (
    (max_block_idx - first_block_idx + 1) * block_size - prompt_offset
)

layer_token_sum = np.zeros((num_layers, total_generated_positions), dtype=np.float64)
layer_token_count = np.zeros((num_layers, total_generated_positions), dtype=np.float64)

for record in records:
    layer_idx = record["layer_idx"]
    block_idx = record["block_idx"]
    token_cos = np.array(record["token_cosine"], dtype=np.float64)

    start_token_offset = 0
    if block_idx == first_block_idx:
        start_token_offset = prompt_offset

    for token_offset in range(start_token_offset, block_size):
        global_token_idx = (
            (block_idx - first_block_idx) * block_size
            + token_offset
            - prompt_offset
        )

        layer_token_sum[layer_idx, global_token_idx] += token_cos[token_offset]
        layer_token_count[layer_idx, global_token_idx] += 1.0

layer_token_mean = layer_token_sum / np.maximum(layer_token_count, 1.0)


plt.figure(figsize=(14, 6))
plt.imshow(layer_token_mean, aspect="auto", origin="lower", cmap="viridis", vmin=0.8, vmax=1.0)
plt.colorbar(label="Mean cosine similarity")
plt.xlabel("Generated token position")
plt.ylabel("Layer")
plt.title("One-sample token similarity heatmap")
plt.tight_layout()
plt.savefig(OUT_DIR / "one_sample_token_heatmap.png", dpi=200)
plt.close()


print("Saved:")
print(OUT_DIR / "one_sample_layer_heatmap.png")
print(OUT_DIR / "one_sample_token_heatmap.png")
