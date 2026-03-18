import json
import os
import torch
import torch.nn.functional as F
import numpy as np


class BaselineTraceRecorder:
    """Collects per-sample decode traces used by the baseline plotting scripts."""

    def __init__(self, save_dir="baseline_logs"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.next_sample_idx = 0
        # Keep the previous denoising-step activations so we can compare step t to step t-1.
        self.prev_ln_hidden = {}
        self.prev_attn_output = {}
        self.prev_ffn_output = {}
        self.prev_attn_weights = {}
        self.prev_ffn_temp = {}
        self.hidden_records = []
        self.attn_records = []
        self.ffn_records = []
        self.ffn_temp_records = []
        self.attn_weight_records = []
        self.attn_range_records = []
        # Part C plots attention-weight magnitude on a fixed log-spaced range.
        self.attn_range_edges = np.logspace(-4, 0, 41)
        self.prompt_len = 0
        self.sample_idx = 0

    def start_new_sample(self, prompt_len):
        # Reset recorder state so each JSON file contains only one sample's traces.
        self.sample_idx = self.next_sample_idx
        self.next_sample_idx += 1
        self.prompt_len = int(prompt_len)
        self.prev_ln_hidden = {}
        self.prev_attn_output = {}
        self.prev_ffn_output = {}
        self.prev_attn_weights = {}
        self.prev_ffn_temp = {}
        self.hidden_records = []
        self.attn_records = []
        self.ffn_records = []
        self.ffn_temp_records = []
        self.attn_weight_records = []
        self.attn_range_records = []

    def _should_log(self, trace_context):
        # Only log the appropriate decode steps.
        if trace_context is None:
            return False
        if trace_context.get("phase") != "decode":
            return False
        if trace_context.get("call_type") != "denoise":
            return False
        return True

    def record_ln_hidden(self, layer_idx, ln_hidden, trace_context):
        # Track how the normalized layer input changes across adjacent denoising steps.
        if not self._should_log(trace_context):
            return

        block_idx = int(trace_context["block_idx"])
        step_idx = int(trace_context["step_idx"])

        current = ln_hidden[0].detach().float().cpu()
        key = (block_idx, int(layer_idx))

        if key in self.prev_ln_hidden:
            prev = self.prev_ln_hidden[key]
            token_cos = F.cosine_similarity(current, prev, dim=-1)
            self.hidden_records.append(
                {
                    "block_idx": block_idx,
                    "step_idx": step_idx,
                    "layer_idx": int(layer_idx),
                    "token_cosine": token_cos.tolist(),
                    "mean_cosine": float(token_cos.mean().item()),
                    "max_cosine": float(token_cos.max().item()),
                    "min_cosine": float(token_cos.min().item()),
                }
            )
        self.prev_ln_hidden[key] = current
    
    def record_attn_output(self, layer_idx, attn_output, trace_context):
        # Compare the attention module output across adjacent denoising steps.
        if not self._should_log(trace_context):
            return

        block_idx = int(trace_context["block_idx"])
        step_idx = int(trace_context["step_idx"])

        current = attn_output[0].detach().float().cpu()
        key = (block_idx, int(layer_idx))
        if key in self.prev_attn_output:
            prev = self.prev_attn_output[key]
            token_cos = F.cosine_similarity(current, prev, dim=-1)
            self.attn_records.append(
                {
                    "block_idx": block_idx,
                    "step_idx": step_idx,
                    "layer_idx": int(layer_idx),
                    "token_cosine": token_cos.tolist(),
                    "mean_cosine": float(token_cos.mean().item()),
                    "max_cosine": float(token_cos.max().item()),
                    "min_cosine": float(token_cos.min().item()),
                }
            )
        self.prev_attn_output[key] = current
    
    def record_ffn_output(self, layer_idx, ffn_output, trace_context):
        # Track the final FFN output similarity across adjacent denoising steps.
        if not self._should_log(trace_context):
            return

        block_idx = int(trace_context["block_idx"])
        step_idx = int(trace_context["step_idx"])

        current = ffn_output[0].detach().float().cpu()
        key = (block_idx, int(layer_idx))

        if key in self.prev_ffn_output:
            prev = self.prev_ffn_output[key]
            token_cos = F.cosine_similarity(current, prev, dim=-1)
            self.ffn_records.append(
                {
                    "block_idx": block_idx,
                    "step_idx": step_idx,
                    "layer_idx": int(layer_idx),
                    "token_cosine": token_cos.tolist(),
                    "mean_cosine": float(token_cos.mean().item()),
                    "max_cosine": float(token_cos.max().item()),
                    "min_cosine": float(token_cos.min().item()),
                }
            )
        self.prev_ffn_output[key] = current
    
    def record_attn_weights(self, layer_idx, attn_weights, trace_context):
        # Record cosine-similarity summary for the attention-weight tensor across adjacent denoising steps.
        if not self._should_log(trace_context):
            return
        
        if attn_weights is None:
            return

        block_idx = int(trace_context["block_idx"])
        step_idx = int(trace_context["step_idx"])

        current = attn_weights[0].detach().float().cpu().reshape(-1)
        current = torch.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
        key = (block_idx, int(layer_idx))

        if key in self.prev_attn_weights:
            prev = self.prev_attn_weights[key]
            prev = torch.nan_to_num(prev, nan=0.0, posinf=0.0, neginf=0.0)

            current_norm = torch.norm(current)
            prev_norm = torch.norm(prev)

            if current_norm.item() == 0.0 or prev_norm.item() == 0.0:
                cosine_value = 0.0
            else:
                cosine = F.cosine_similarity(
                    current.unsqueeze(0),
                    prev.unsqueeze(0),
                    dim=-1,
                )
                cosine_value = float(cosine.item())

                if torch.isnan(torch.tensor(cosine_value)):
                    cosine_value = 0.0

            self.attn_weight_records.append(
                {
                    "block_idx": block_idx,
                    "step_idx": step_idx,
                    "layer_idx": int(layer_idx),
                    "cosine_similarity": cosine_value,
                }
            )
        self.prev_attn_weights[key] = current
    
    def record_attn_weight_range(self, layer_idx, attn_weights, trace_context):
        # Bucket attention-weight magnitudes so Part C can plot their log-scale value distribution.
        if not self._should_log(trace_context):
            return

        if attn_weights is None:
            return

        block_idx = int(trace_context["block_idx"])
        step_idx = int(trace_context["step_idx"])

        values = attn_weights[0].detach().float().cpu().reshape(-1)
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = values.numpy()

        num_below_range = int(np.sum(values < 1e-4))
        num_above_range = int(np.sum(values > 1.0))

        # Keep the plotted histogram focused on [1e-4, 1] and track the overflow separately.
        valid_values = values[(values >= 1e-4) & (values <= 1.0)]

        bin_counts, _ = np.histogram(valid_values, bins=self.attn_range_edges)

        self.attn_range_records.append(
            {
                "block_idx": block_idx,
                "step_idx": step_idx,
                "layer_idx": int(layer_idx),
                "bin_counts": bin_counts.tolist(),
                "total_count": int(len(values)),
                "num_in_range": int(len(valid_values)),
                "num_below_range": num_below_range,
                "num_above_range": num_above_range,
            }
        )

    def record_ffn_temp(self, layer_idx, ffn_temp, trace_context):
        # Compare the FFN intermediate used in Part C (before the down projection is applied) across adjacent denoising steps.
        if not self._should_log(trace_context):
            return

        block_idx = int(trace_context["block_idx"])
        step_idx = int(trace_context["step_idx"])

        current = ffn_temp[0].detach().float().cpu()
        key = (block_idx, int(layer_idx))

        if key in self.prev_ffn_temp:
            prev = self.prev_ffn_temp[key]
            token_cos = F.cosine_similarity(current, prev, dim=-1)

            self.ffn_temp_records.append(
                {
                    "block_idx": block_idx,
                    "step_idx": step_idx,
                    "layer_idx": int(layer_idx),
                    "token_cosine": token_cos.tolist(),
                    "mean_cosine": float(token_cos.mean().item()),
                    "max_cosine": float(token_cos.max().item()),
                    "min_cosine": float(token_cos.min().item()),
                }
            )
        self.prev_ffn_temp[key] = current
 

    def save_current_sample(self):
        # Each sample is saved as a standalone JSON file for the plotting scripts.
        out_path = os.path.join(self.save_dir, f"sample_{self.sample_idx:04d}.json")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "sample_idx": self.sample_idx,
                    "prompt_len": self.prompt_len,
                    # Uncomment the record families below needed for a specific baseline plot.
                    # "hidden_records": self.hidden_records,
                    # "attn_records": self.attn_records,
                    # "ffn_records": self.ffn_records,
                    # "attn_weight_records": self.attn_weight_records,
                    # "attn_range_records": self.attn_range_records,
                    # "ffn_temp_records": self.ffn_temp_records,
                },
                f,
                indent=2,
            )
