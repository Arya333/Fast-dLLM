import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOG_DIR = Path("baseline_logs_100_haf")
OUT_DIR = Path("plotting")
OUT_DIR.mkdir(exist_ok=True)

# If there are too many points, randomly sample for a cleaner figure
MAX_POINTS_TO_PLOT = 50000
RNG = np.random.default_rng(0)


def load_all_points(log_dir):
    json_paths = sorted(log_dir.glob("sample_*.json"))
    if len(json_paths) == 0:
        raise ValueError(f"No sample json files found in {log_dir}")

    hidden_points = {}
    attn_points = {}
    ffn_points = {}

    for path in json_paths:
        with open(path, "r") as f:
            data = json.load(f)

        sample_idx = data["sample_idx"]

        for record in data["hidden_records"]:
            key = (
                sample_idx,
                record["block_idx"],
                record["step_idx"],
                record["layer_idx"],
            )
            hidden_points[key] = record["mean_cosine"]

        for record in data["attn_records"]:
            key = (
                sample_idx,
                record["block_idx"],
                record["step_idx"],
                record["layer_idx"],
            )
            attn_points[key] = record["mean_cosine"]

        for record in data["ffn_records"]:
            key = (
                sample_idx,
                record["block_idx"],
                record["step_idx"],
                record["layer_idx"],
            )
            ffn_points[key] = record["mean_cosine"]

    common_keys = sorted(set(hidden_points) & set(attn_points) & set(ffn_points))
    if len(common_keys) == 0:
        raise ValueError("No matched hidden/attention/FFN records were found.")

    hidden = np.array([hidden_points[key] for key in common_keys], dtype=np.float64)
    attn = np.array([attn_points[key] for key in common_keys], dtype=np.float64)
    ffn = np.array([ffn_points[key] for key in common_keys], dtype=np.float64)
    return hidden, attn, ffn


def maybe_subsample(x, y, max_points):
    if len(x) <= max_points:
        return x, y

    keep_idx = RNG.choice(len(x), size=max_points, replace=False)
    return x[keep_idx], y[keep_idx]


def pearson_corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def make_scatter(x, y, x_label, y_label, title, out_path):
    x_plot, y_plot = maybe_subsample(x, y, MAX_POINTS_TO_PLOT)
    rho = pearson_corr(x, y)

    plt.figure(figsize=(6, 6))
    plt.scatter(x_plot, y_plot, s=1, alpha=0.15, color="tab:blue", edgecolors="none")
    plt.plot([0.85, 1.0], [0.85, 1.0], linestyle="--", color="red", linewidth=1)
    plt.xlim(0.85, 1.0)
    plt.ylim(0.85, 1.0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title}\nr = {rho:.3f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return rho, len(x), len(x_plot)


hidden, attn, ffn = load_all_points(LOG_DIR)

rho_hidden_attn, total_points, plotted_points = make_scatter(
    hidden,
    attn,
    x_label="Similarity of hidden state",
    y_label="Similarity of attention output",
    title="Hidden vs Attention Output",
    out_path=OUT_DIR / "hidden_vs_attn_scatter.png",
)

rho_hidden_ffn, _, _ = make_scatter(
    hidden,
    ffn,
    x_label="Similarity of hidden state",
    y_label="Similarity of FFN output",
    title="Hidden vs FFN Output",
    out_path=OUT_DIR / "hidden_vs_ffn_scatter.png",
)

print("Saved:")
print(OUT_DIR / "hidden_vs_attn_scatter.png")
print(OUT_DIR / "hidden_vs_ffn_scatter.png")
print("Total matched points:", total_points)
print("Plotted points:", plotted_points)
print("Correlation hidden vs attention:", round(rho_hidden_attn, 4))
print("Correlation hidden vs FFN:", round(rho_hidden_ffn, 4))
