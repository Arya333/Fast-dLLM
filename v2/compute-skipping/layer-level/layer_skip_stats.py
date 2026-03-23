# Saves the per-layer skip statistics that are used for FLOPs calculations and summary tables.
import json
import os


class LayerSkipStatsRecorder:
    def __init__(self, save_dir="compute-skipping/logs"):
        # Keep one log folder and one in-memory record buffer for the current sample.
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.next_sample_idx = 0
        self.sample_idx = 0
        self.prompt_len = 0

        self.current_sample_total_denoising_steps = 0
        self.layer_step_records = []

    def start_new_sample(self, prompt_len):
        # Reset the recorder so each sample is written to its own JSON file.
        self.sample_idx = self.next_sample_idx
        self.next_sample_idx += 1
        self.prompt_len = int(prompt_len)

        self.current_sample_total_denoising_steps = 0
        self.layer_step_records = []

    def set_sample_total_denoising_steps(self, total_denoising_steps):
        # Save the outer decode-step count used later in the summary table.
        self.current_sample_total_denoising_steps = int(total_denoising_steps)

    def record_layer_step(
        self,
        block_idx,
        step_idx,
        layer_idx,
        num_total_tokens,
        layer_similarity=None,
        token_cosine=None,
        skipped=False,
        threshold=None,
        aggregation=None,
    ):
        # Save one layer-step record with both the skip decision and summary similarity stats.
        num_active_tokens = 0 if skipped else int(num_total_tokens)
        record = {
            "block_idx": int(block_idx),
            "step_idx": int(step_idx),
            "layer_idx": int(layer_idx),
            "num_total_tokens": int(num_total_tokens),
            "num_active_tokens": int(num_active_tokens),
            "num_reused_tokens": int(num_total_tokens - num_active_tokens),
            "skipped": bool(skipped),
        }

        if aggregation is not None:
            record["aggregation"] = aggregation

        if layer_similarity is not None:
            similarity_value = float(layer_similarity.item() if hasattr(layer_similarity, "item") else layer_similarity)
            record["layer_similarity"] = similarity_value

            if threshold is not None:
                record["threshold"] = float(threshold)
                record["above_threshold"] = bool(similarity_value >= float(threshold))

        if token_cosine is not None:
            token_cosine_cpu = token_cosine[0].detach().float().cpu()
            record["mean_token_cosine"] = float(token_cosine_cpu.mean().item())
            record["min_token_cosine"] = float(token_cosine_cpu.min().item())
            record["max_token_cosine"] = float(token_cosine_cpu.max().item())

        self.layer_step_records.append(record)

    def save_current_sample(self):
        # Write the completed sample summary in the format expected by the plotting scripts.
        out_path = os.path.join(self.save_dir, f"sample_{self.sample_idx:04d}.json")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "sample_idx": self.sample_idx,
                    "prompt_len": self.prompt_len,
                    "total_denoising_steps": self.current_sample_total_denoising_steps,
                    "layer_step_records": self.layer_step_records,
                },
                f,
                indent=2,
            )
