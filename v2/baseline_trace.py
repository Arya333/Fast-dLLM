import json
import os
import torch
import torch.nn.functional as F


class BaselineTraceRecorder:
    def __init__(self, save_dir="baseline_logs"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.next_sample_idx = 0
        self.prev_ln_hidden = {} # a dictionary that stores the previous step’s post-layernorm hidden states, so we can compare step t to step t-1
        self.prev_attn_output = {}
        self.prev_ffn_output = {}
        self.hidden_records = [] # list of similarity results to save for the current sample
        self.attn_records = []
        self.ffn_records = []
        self.prompt_len = 0
        self.sample_idx = 0

    def start_new_sample(self, prompt_len):
        self.sample_idx = self.next_sample_idx
        self.next_sample_idx += 1
        self.prompt_len = int(prompt_len)
        self.prev_ln_hidden = {}
        self.prev_attn_output = {}
        self.prev_ffn_output = {}
        self.hidden_records = []
        self.attn_records = []
        self.ffn_records = []

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

    def save_current_sample(self):
        out_path = os.path.join(self.save_dir, f"sample_{self.sample_idx:04d}.json")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "sample_idx": self.sample_idx,
                    "prompt_len": self.prompt_len,
                    "hidden_records": self.hidden_records,
                    "attn_records": self.attn_records,
                    "ffn_records": self.ffn_records,
                },
                f,
                indent=2,
            )
