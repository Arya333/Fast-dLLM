import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PERCENTILES = [1, 5, 25, 50, 75, 95, 99]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot diagnostic cosine histograms from token-skip logs."
    )
    parser.add_argument(
        "--log-dirs",
        nargs="+",
        default=[
            "compute-skipping/logs/token_threshold_0.995",
            "compute-skipping/logs/token_threshold_0.96",
        ],
        help="One or more log directories to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional display labels for the log directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("compute-skipping/plotting/token_cosine_diagnostics.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def load_run(log_dir):
    log_path = Path(log_dir)
    json_paths = sorted(log_path.glob("sample_*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No sample JSONs found in {log_dir}")

    mean_values = []
    min_values = []
    max_values = []
    below_threshold_counts = []
    active_counts = []
    total_token_counts = []

    for path in json_paths:
        with open(path, "r") as f:
            data = json.load(f)

        for record in data["layer_step_records"]:
            if "mean_cosine" not in record:
                continue

            mean_values.append(float(record["mean_cosine"]))
            min_values.append(float(record["min_cosine"]))
            max_values.append(float(record["max_cosine"]))
            below_threshold_counts.append(float(record.get("num_tokens_below_threshold", 0)))
            active_counts.append(float(record["num_active_tokens"]))
            total_token_counts.append(float(record["num_total_tokens"]))

    if not mean_values:
        raise ValueError(
            f"No cosine debug fields found in {log_dir}. "
            "These logs need mean_cosine/min_cosine/max_cosine to be present."
        )

    return {
        "log_dir": str(log_path),
        "mean": np.array(mean_values, dtype=np.float64),
        "min": np.array(min_values, dtype=np.float64),
        "max": np.array(max_values, dtype=np.float64),
        "below_threshold": np.array(below_threshold_counts, dtype=np.float64),
        "active": np.array(active_counts, dtype=np.float64),
        "total": np.array(total_token_counts, dtype=np.float64),
    }


def percentile_summary(values):
    return {f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES}


def print_summary(label, run):
    print(f"\n{label}")
    print(f"  log_dir: {run['log_dir']}")
    print(f"  num_records_with_cosines: {len(run['mean'])}")
    print("  mean_cosine percentiles:", percentile_summary(run["mean"]))
    print("  min_cosine percentiles: ", percentile_summary(run["min"]))
    print("  max_cosine percentiles: ", percentile_summary(run["max"]))
    print(
        "  avg tokens below threshold per record:",
        float(run["below_threshold"].mean()),
    )
    print(
        "  avg active tokens per record:",
        float(run["active"].mean()),
    )
    print(
        "  avg active token fraction:",
        float(run["active"].sum() / run["total"].sum()),
    )


def add_hist(ax, runs, key, title):
    bins = np.linspace(0.0, 1.0005, 120)
    for label, run in runs:
        ax.hist(
            run[key],
            bins=bins,
            alpha=0.45,
            label=label,
            density=False,
        )

    ax.set_title(title)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Occurrence")
    ax.set_xlim(0.0, 1.0005)
    ax.grid(True, linestyle="--", alpha=0.25)


def main():
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.log_dirs):
        raise ValueError("--labels must have the same length as --log-dirs")

    labels = args.labels
    if labels is None:
        labels = [Path(log_dir).name for log_dir in args.log_dirs]

    runs = []
    for label, log_dir in zip(labels, args.log_dirs):
        run = load_run(log_dir)
        runs.append((label, run))
        print_summary(label, run)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    add_hist(axes[0], runs, "mean", "Record-Level Mean Cosine")
    add_hist(axes[1], runs, "min", "Record-Level Min Cosine")
    add_hist(axes[2], runs, "max", "Record-Level Max Cosine")

    axes[2].legend(loc="upper left")
    fig.suptitle("Token-Skip Cosine Diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    print(f"\nSaved histogram figure to {args.output}")
    print(
        "Note: this uses record-level mean/min/max cosine values. "
        "It is not the exact raw token-wise cosine histogram."
    )


if __name__ == "__main__":
    main()
