import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


GROUP_STYLE = {
    "baseline": {
        "legend_label": "Baseline",
        "color": "#2563eb",
        "marker": "*",
        "size": 220,
        "annotation_offset": (10, 8),
        "zorder": 8,
    },
    "token_cosine": {
        "legend_label": "Token-level cosine",
        "color": "#16a34a",
        "marker": "o",
        "size": 95,
        "annotation_offset": (8, 8),
        "zorder": 4,
    },
    "token_topk": {
        "legend_label": "Token-level top-k",
        "color": "#15803d",
        "marker": "^",
        "size": 110,
        "annotation_offset": (8, -14),
        "zorder": 4,
    },
    "layer_avg": {
        "legend_label": "Layer-level avg",
        "color": "#f59e0b",
        "marker": "s",
        "size": 95,
        "annotation_offset": (8, 8),
        "zorder": 3,
    },
    "layer_max": {
        "legend_label": "Layer-level max",
        "color": "#dc2626",
        "marker": "D",
        "size": 95,
        "annotation_offset": (8, -14),
        "zorder": 5,
    },
}

ANNOTATION_OFFSET_OVERRIDES = {
    "baseline": (-30, 12),
    "token_threshold_0.995": (-30, 10),
    "token_threshold_0.99": (-12, -14),
    "token_threshold_0.98": (-10, 18),
    "token_threshold_0.97": (12, -16),
    "token_threshold_0.96": (16, 18),
    "token_topk_25": (8, -16),
    "token_topk_50": (8, -16),
    "layer_avg_0.999": (16, 12),
    "layer_avg_0.995": (16, 4),
    "layer_avg_0.99": (16, -14),
    "layer_avg_0.98": (16, 8),
    "layer_avg_0.97": (8, 10),
    "layer_max_0.999": (10, 16),
    "layer_max_0.995": (10, 4),
    "layer_max_0.99": (10, -10),
    "layer_max_0.98": (10, 28),
    "layer_max_0.97": (10, -24),
}

SETTING_ORDER = [
    "baseline",
    "token_threshold_0.995",
    "token_threshold_0.99",
    "token_threshold_0.98",
    "token_threshold_0.97",
    "token_threshold_0.96",
    "token_topk_25",
    "token_topk_50",
    "layer_avg_0.999",
    "layer_avg_0.995",
    "layer_avg_0.99",
    "layer_avg_0.98",
    "layer_avg_0.97",
    "layer_max_0.999",
    "layer_max_0.995",
    "layer_max_0.99",
    "layer_max_0.98",
    "layer_max_0.97",
]

DEFAULT_MODEL_SPECS = {
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "bd_size": 32,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot accuracy vs FLOPs reduction for baseline/token/layer skipping settings."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("compute-skipping/plotting/overall_results.csv"),
        help="CSV summary file with accuracy, FLOPs reduction, and avg denoising steps.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("compute-skipping/plotting/accuracy_vs_flops.png"),
        help="Output path for the scatter plot.",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=Path("compute-skipping/plotting/avg_denoising_steps_table.png"),
        help="Output path for the average denoising steps table.",
    )
    parser.add_argument(
        "--output-table-tex",
        type=Path,
        default=Path("compute-skipping/plotting/avg_denoising_steps_table.tex"),
        help="Output path for the LaTeX version of the average denoising steps table.",
    )
    parser.add_argument(
        "--output-resolved-csv",
        type=Path,
        default=Path("compute-skipping/plotting/overall_results_resolved.csv"),
        help="Output path for the resolved CSV with accuracy, FLOPs reduction, and avg denoising steps.",
    )
    parser.add_argument(
        "--title",
        default="Accuracy vs FLOPs Reduction",
        help="Plot title.",
    )
    parser.add_argument(
        "--baseline-log-dir",
        type=Path,
        default=Path("compute-skipping/logs/baseline"),
        help="Baseline log directory used to compute FLOPs reduction.",
    )
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_MODEL_SPECS["hidden_size"])
    parser.add_argument("--intermediate-size", type=int, default=DEFAULT_MODEL_SPECS["intermediate_size"])
    parser.add_argument("--num-attention-heads", type=int, default=DEFAULT_MODEL_SPECS["num_attention_heads"])
    parser.add_argument("--num-key-value-heads", type=int, default=DEFAULT_MODEL_SPECS["num_key_value_heads"])
    parser.add_argument("--bd-size", type=int, default=DEFAULT_MODEL_SPECS["bd_size"])
    return parser.parse_args()


def maybe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def to_percent(value):
    if value is None:
        return None
    if value <= 1.0:
        return value * 100.0
    return value


def compute_avg_denoising_steps(log_dir):
    if log_dir is None or str(log_dir).strip() == "":
        return None

    log_path = Path(log_dir)
    json_paths = sorted(log_path.glob("sample_*.json"))
    if not json_paths:
        return None

    total_steps = 0.0
    for path in json_paths:
        with open(path, "r") as f:
            data = json.load(f)
        total_steps += float(data["total_denoising_steps"])

    return total_steps / len(json_paths)


def compute_kv_hidden_size(model_specs):
    return (
        model_specs["hidden_size"]
        * model_specs["num_key_value_heads"]
        // model_specs["num_attention_heads"]
    )


def compute_record_flops(record, model_specs):
    hidden_size = model_specs["hidden_size"]
    intermediate_size = model_specs["intermediate_size"]
    kv_hidden_size = compute_kv_hidden_size(model_specs)
    block_size = model_specs["bd_size"]

    num_total_tokens = int(record["num_total_tokens"])
    num_active_tokens = int(record["num_active_tokens"])
    block_idx = int(record["block_idx"])

    # Current implementation attends from the current block to all previous blocks
    key_length = block_idx * block_size + num_total_tokens

    if num_active_tokens == 0:
        return 0.0

    q_proj_flops = 2.0 * num_active_tokens * hidden_size * hidden_size
    k_proj_flops = 2.0 * num_total_tokens * hidden_size * kv_hidden_size
    v_proj_flops = 2.0 * num_total_tokens * hidden_size * kv_hidden_size
    qk_flops = 2.0 * num_active_tokens * key_length * hidden_size
    av_flops = 2.0 * num_active_tokens * key_length * hidden_size
    o_proj_flops = 2.0 * num_active_tokens * hidden_size * hidden_size

    mlp_flops = 6.0 * num_active_tokens * hidden_size * intermediate_size

    return (
        q_proj_flops
        + k_proj_flops
        + v_proj_flops
        + qk_flops
        + av_flops
        + o_proj_flops
        + mlp_flops
    )


def compute_total_flops(log_dir, model_specs):
    if log_dir is None or str(log_dir).strip() == "":
        return None

    log_path = Path(log_dir)
    json_paths = sorted(log_path.glob("sample_*.json"))
    if not json_paths:
        return None

    total_flops = 0.0
    for path in json_paths:
        with open(path, "r") as f:
            data = json.load(f)
        for record in data["layer_step_records"]:
            total_flops += compute_record_flops(record, model_specs)

    return total_flops


def load_rows(summary_path, baseline_log_dir, model_specs):
    if not summary_path.exists():
        template_path = summary_path.with_name("overall_results_template.csv")
        raise FileNotFoundError(
            f"Could not find {summary_path}. Start from {template_path} and fill in your metrics."
        )

    rows = []
    skipped_settings = []
    flops_cache = {}

    baseline_total_flops = compute_total_flops(baseline_log_dir, model_specs)

    with open(summary_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            setting_id = (raw_row.get("setting_id") or "").strip()
            if setting_id == "":
                continue

            group = (raw_row.get("group") or "").strip()
            if group not in GROUP_STYLE:
                raise ValueError(
                    f"Unknown group '{group}' for setting '{setting_id}'. "
                    f"Expected one of: {sorted(GROUP_STYLE)}"
                )

            accuracy = to_percent(maybe_float(raw_row.get("accuracy")))
            flop_reduction_pct = to_percent(maybe_float(raw_row.get("flop_reduction_pct")))
            avg_denoising_steps = maybe_float(raw_row.get("avg_denoising_steps"))
            log_dir = (raw_row.get("log_dir") or "").strip()
            effective_log_dir = log_dir
            if effective_log_dir == "" and group == "baseline":
                effective_log_dir = str(baseline_log_dir)

            if avg_denoising_steps is None and effective_log_dir:
                avg_denoising_steps = compute_avg_denoising_steps(effective_log_dir)

            if flop_reduction_pct is None:
                if group == "baseline":
                    flop_reduction_pct = 0.0
                elif effective_log_dir and baseline_total_flops is not None and baseline_total_flops > 0:
                    if effective_log_dir not in flops_cache:
                        flops_cache[effective_log_dir] = compute_total_flops(effective_log_dir, model_specs)
                    setting_total_flops = flops_cache[effective_log_dir]
                    if setting_total_flops is not None:
                        flop_reduction_pct = 100.0 * (1.0 - setting_total_flops / baseline_total_flops)

            row = {
                "setting_id": setting_id,
                "group": group,
                "display_name": (raw_row.get("display_name") or setting_id).strip(),
                "point_label": (raw_row.get("point_label") or setting_id).strip(),
                "accuracy_pct": accuracy,
                "flop_reduction_pct": flop_reduction_pct,
                "avg_denoising_steps": avg_denoising_steps,
                "log_dir": effective_log_dir,
            }

            if row["accuracy_pct"] is None or row["flop_reduction_pct"] is None:
                skipped_settings.append(setting_id)
                continue

            rows.append(row)

    rows.sort(key=lambda row: SETTING_ORDER.index(row["setting_id"]) if row["setting_id"] in SETTING_ORDER else len(SETTING_ORDER))
    return rows, skipped_settings


def make_plot(rows, output_path, title):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.3))

    for group_name, style in GROUP_STYLE.items():
        group_rows = [row for row in rows if row["group"] == group_name]
        if not group_rows:
            continue

        xs = [row["flop_reduction_pct"] for row in group_rows]
        ys = [row["accuracy_pct"] for row in group_rows]

        ax.scatter(
            xs,
            ys,
            s=style["size"],
            c=style["color"],
            marker=style["marker"],
            edgecolors="black",
            linewidths=0.8,
            alpha=0.92,
            label=style["legend_label"],
            zorder=style.get("zorder", 3),
        )

        for row in group_rows:
            dx, dy = ANNOTATION_OFFSET_OVERRIDES.get(
                row["setting_id"], style["annotation_offset"]
            )
            ax.annotate(
                row["point_label"],
                xy=(row["flop_reduction_pct"], row["accuracy_pct"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.5,
                color=style["color"],
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
                zorder=style.get("zorder", 3) + 1,
            )

    x_values = [row["flop_reduction_pct"] for row in rows]
    y_values = [row["accuracy_pct"] for row in rows]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    x_pad = max(2.0, (x_max - x_min) * 0.12 if x_max != x_min else 2.0)
    y_pad = max(0.8, (y_max - y_min) * 0.18 if y_max != y_min else 0.8)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, min(100.0, y_max + y_pad))
    ax.set_xlabel("FLOPs Reduction Compared to Baseline (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.axvline(0.0, color="#9ca3af", linestyle=":", linewidth=1.0, zorder=1)

    ax.annotate(
        "Better",
        xy=(0.86, 0.79),
        xycoords="axes fraction",
        xytext=(0.70, 0.64),
        textcoords="axes fraction",
        color="#1d4ed8",
        fontsize=10,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "simple",
            "facecolor": "#2563eb",
            "edgecolor": "#2563eb",
            "alpha": 0.8,
        },
    )

    legend = ax.legend(loc="lower left", frameon=True)
    legend.get_frame().set_alpha(0.92)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_table(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_height = max(4.8, 0.34 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    ax.axis("off")

    cell_text = []
    for row in rows:
        avg_steps = row["avg_denoising_steps"]
        avg_steps_text = "" if avg_steps is None else f"{avg_steps:.2f}"
        cell_text.append([row["display_name"], avg_steps_text])

    table = ax.table(
        cellText=cell_text,
        colLabels=["Setting", "Avg. denoising steps"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.67, 0.33],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.26)

    all_cells = table.get_celld()
    for (_, _), cell in all_cells.items():
        cell.set_facecolor("white")
        cell.set_edgecolor("white")
        cell.set_linewidth(0.0)
        cell.visible_edges = ""
        cell.get_text().set_fontfamily("DejaVu Serif")

    for col_idx in range(2):
        header = table[(0, col_idx)]
        header.visible_edges = "TB"
        header.set_edgecolor("black")
        header.set_linewidth(1.2)
        header.get_text().set_weight("bold")
        header.get_text().set_fontfamily("DejaVu Serif")

    for row_idx in range(1, len(rows) + 1):
        table[(row_idx, 0)].get_text().set_ha("left")
        table[(row_idx, 1)].get_text().set_ha("right")

    section_start_ids = {
        "token_threshold_0.995",
        "layer_avg_0.999",
        "layer_max_0.999",
    }
    for row_idx, row in enumerate(rows, start=1):
        if row["setting_id"] in section_start_ids:
            for col_idx in range(2):
                cell = table[(row_idx, col_idx)]
                cell.visible_edges = "T"
                cell.set_edgecolor("black")
                cell.set_linewidth(0.8)

    last_row_idx = len(rows)
    for col_idx in range(2):
        cell = table[(last_row_idx, col_idx)]
        cell.visible_edges = "".join(sorted(set(cell.visible_edges + "B")))
        cell.set_edgecolor("black")
        cell.set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_resolved_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "setting_id",
        "group",
        "display_name",
        "point_label",
        "accuracy_pct",
        "flop_reduction_pct",
        "avg_denoising_steps",
        "log_dir",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "setting_id": row["setting_id"],
                    "group": row["group"],
                    "display_name": row["display_name"],
                    "point_label": row["point_label"],
                    "accuracy_pct": f"{row['accuracy_pct']:.2f}",
                    "flop_reduction_pct": f"{row['flop_reduction_pct']:.2f}",
                    "avg_denoising_steps": (
                        "" if row["avg_denoising_steps"] is None else f"{row['avg_denoising_steps']:.2f}"
                    ),
                    "log_dir": row["log_dir"],
                }
            )


def save_latex_table(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Setting & Avg. denoising steps \\\\",
        "\\midrule",
    ]

    section_start_ids = {
        "token_threshold_0.995",
        "layer_avg_0.999",
        "layer_max_0.999",
    }

    for row in rows:
        if row["setting_id"] in section_start_ids:
            lines.append("\\midrule")
        avg_steps_text = "N/A" if row["avg_denoising_steps"] is None else f"{row['avg_denoising_steps']:.2f}"
        display_name = row["display_name"].replace("%", "\\%")
        lines.append(f"{display_name} & {avg_steps_text} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")

    output_path.write_text("\n".join(lines))


def main():
    args = parse_args()
    model_specs = {
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "bd_size": args.bd_size,
    }
    rows, skipped_settings = load_rows(args.summary, args.baseline_log_dir, model_specs)

    if not rows:
        raise ValueError(
            "No complete rows found in the summary file. Fill in at least accuracy and flop_reduction_pct."
        )

    make_plot(rows, args.output_plot, args.title)
    make_table(rows, args.output_table)
    save_latex_table(rows, args.output_table_tex)
    save_resolved_csv(rows, args.output_resolved_csv)

    print(f"Saved plot to {args.output_plot}")
    print(f"Saved table to {args.output_table}")
    print(f"Saved LaTeX table to {args.output_table_tex}")
    print(f"Saved resolved CSV to {args.output_resolved_csv}")
    print("Loaded settings:")
    for row in rows:
        avg_steps_text = "N/A" if row["avg_denoising_steps"] is None else f"{row['avg_denoising_steps']:.2f}"
        print(
            f"  - {row['setting_id']}: accuracy={row['accuracy_pct']:.2f}%, "
            f"flops_reduction={row['flop_reduction_pct']:.2f}%, avg_steps={avg_steps_text}"
        )
    if skipped_settings:
        print("Skipped incomplete settings:")
        for setting_id in skipped_settings:
            print(f"  - {setting_id}")


if __name__ == "__main__":
    main()
