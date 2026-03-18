import torch
import math

from token_skip_policy import TokenSkipPolicy


class TokenSkipManager:
    def __init__(self, config, stats_recorder=None):
        self.config = config
        self.policy = TokenSkipPolicy(config)
        self.stats_recorder = stats_recorder

        self.prev_ln_hidden = {}
        self.prev_layer_output = {}

    def start_new_sample(self):
        self.prev_ln_hidden = {}
        self.prev_layer_output = {}

    def _should_skip(self, trace_context):
        if not self.config.enabled:
            return False
        if trace_context is None:
            return False
        if trace_context.get("phase") != "decode":
            return False
        if trace_context.get("call_type") != "denoise":
            return False
        if int(trace_context["step_idx"]) == 0:
            return False
        return True

    def build_layer_skip_plan(self, layer_idx, ln_hidden, trace_context):
        if ln_hidden.shape[0] != 1:
            raise ValueError("Token skipping currently expects batch_size=1")

        block_idx = int(trace_context["block_idx"])
        step_idx = int(trace_context["step_idx"])
        key = (block_idx, int(layer_idx))

        num_total_tokens = int(ln_hidden.shape[1])
        all_active_mask = torch.ones(
            (ln_hidden.shape[0], ln_hidden.shape[1]),
            dtype=torch.bool,
            device=ln_hidden.device,
        )
        all_reuse_mask = torch.zeros_like(all_active_mask)

        prev_ln_hidden = self.prev_ln_hidden.get(key)
        prev_layer_output = self.prev_layer_output.get(key)

        if (not self._should_skip(trace_context)) or prev_ln_hidden is None or prev_layer_output is None:
            return {
                "block_idx": block_idx,
                "step_idx": step_idx,
                "key": key,
                "num_total_tokens": num_total_tokens,
                "num_active_tokens": num_total_tokens,
                "token_cosine": None,
                "active_mask": all_active_mask,
                "reuse_mask": all_reuse_mask,
                "mixed_ln_hidden": ln_hidden,
                "prev_ln_hidden": prev_ln_hidden,
                "prev_layer_output": prev_layer_output,
                "did_skip": False,
            }

        prev_ln_hidden = prev_ln_hidden.to(device=ln_hidden.device, dtype=ln_hidden.dtype)
        prev_layer_output = prev_layer_output.to(device=ln_hidden.device, dtype=ln_hidden.dtype)

        masks = self.policy.build_masks(ln_hidden, prev_ln_hidden)
        active_mask = masks["active_mask"]
        reuse_mask = masks["reuse_mask"]

        if self.config.mode == "threshold":
            expected_active = int((masks["token_cosine"] < self.config.threshold).sum().item())
            actual_active = int(masks["active_mask"].sum().item())
            if expected_active != actual_active:
                raise ValueError("Threshold token skip mask does not match cosine rule")
        else:
            expected_active = max(
                1,
                math.ceil(masks["token_cosine"].shape[1] * self.config.topk_percent / 100.0),
            )
            actual_active = int(masks["active_mask"].sum().item())
            if expected_active != actual_active:
                raise ValueError("Top-k token skip mask does not match requested k")


        mixed_ln_hidden = ln_hidden.clone()
        mixed_ln_hidden[0, reuse_mask[0]] = prev_ln_hidden[0, reuse_mask[0]]

        return {
            "block_idx": block_idx,
            "step_idx": step_idx,
            "key": key,
            "num_total_tokens": num_total_tokens,
            "num_active_tokens": int(active_mask.sum().item()),
            "token_cosine": masks["token_cosine"],
            "active_mask": active_mask,
            "reuse_mask": reuse_mask,
            "mixed_ln_hidden": mixed_ln_hidden,
            "prev_ln_hidden": prev_ln_hidden,
            "prev_layer_output": prev_layer_output,
            "did_skip": True,
        }
    
    def finish_layer(self, layer_idx, ln_hidden, layer_output, trace_context, skip_plan):
        key = skip_plan["key"]

        self.prev_ln_hidden[key] = ln_hidden.detach().cpu()
        self.prev_layer_output[key] = layer_output.detach().cpu()

        if self.stats_recorder is not None:
            threshold = self.config.threshold if self.config.mode == "threshold" else None
            self.stats_recorder.record_layer_step(
                block_idx=skip_plan["block_idx"],
                step_idx=skip_plan["step_idx"],
                layer_idx=layer_idx,
                num_total_tokens=skip_plan["num_total_tokens"],
                num_active_tokens=skip_plan["num_active_tokens"],
                token_cosine=skip_plan["token_cosine"],
                active_mask=skip_plan["active_mask"],
                threshold=threshold,
            )

        return layer_output
