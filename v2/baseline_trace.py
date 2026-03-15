import json
import os
import torch
import torch.nn.functional as F
import numpy as np


class BaselineTraceRecorder:
    def __init__(self, save_dir="baseline_logs"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.next_sample_idx = 0
        self.prev_ln_hidden = {} # a dictionary that stores the previous step’s post-layernorm hidden states, so we can compare step t to step t-1
        self.prev_attn_output = {}
        self.prev_ffn_output = {}
        self.prev_attn_weights = {}
        self.prev_ffn_temp = {}
        self.hidden_records = [] # list of similarity results to save for the current sample
        self.attn_records = []
        self.ffn_records = []
        self.ffn_temp_records = []
        self.attn_weight_records = []
        self.attn_range_records = []
        self.attn_range_edges = np.logspace(-4, 0, 41) # 40 log-scale bins from 1e-4 to 1
        self.prompt_len = 0
        self.sample_idx = 0

    def start_new_sample(self, prompt_len):
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
        if trace_context is None:
            return False
        if trace_context.get("phase") != "decode":
            return False
        if trace_context.get("call_type") != "denoise":
            return False
        return True

    def record_ln_hidden(self, layer_idx, ln_hidden, trace_context):
        # Only compare true denoising steps, not prefill / final block commit
        if not self._should_log(trace_context):
            return

        block_idx = int(trace_context["block_idx"]) # which output block we are denoising right now
        step_idx = int(trace_context["step_idx"]) # which denoising iteration inside that block we are on

        current = ln_hidden[0].detach().float().cpu()  # start with batch_size = 1
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

        # Keep only values inside the plotted range
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
        out_path = os.path.join(self.save_dir, f"sample_{self.sample_idx:04d}.json")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "sample_idx": self.sample_idx,
                    "prompt_len": self.prompt_len,
                    # "hidden_records": self.hidden_records,
                    # "attn_records": self.attn_records,
                    # "ffn_records": self.ffn_records,
                    # "attn_weight_records": self.attn_weight_records,
                    # "attn_range_records": self.attn_range_records,
                    "ffn_temp_records": self.ffn_temp_records,
                },
                f,
                indent=2,
            )
