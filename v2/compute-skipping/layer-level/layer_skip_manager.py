# Tracks per-sample layer state and applies the layer-level skipping decision during decoding.
import torch

from layer_skip_policy import LayerSkipPolicy


class LayerSkipManager:
    def __init__(self, config, stats_recorder=None):
        # Keep the policy, logger, and previous-step activations needed for layer reuse.
        self.config = config
        self.policy = LayerSkipPolicy(config)
        self.stats_recorder = stats_recorder

        self.prev_ln_hidden = {}
        self.prev_layer_output = {}

    def start_new_sample(self):
        # Reset cached activations so each sample starts with a clean layer-skip state.
        self.prev_ln_hidden = {}
        self.prev_layer_output = {}

    def _should_skip(self, trace_context):
        # Only run layer skipping on real decode denoising steps after step 0.
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
        # Build the reuse decision for one layer at one denoising step.
        if ln_hidden.shape[0] != 1:
            raise ValueError("Layer skipping currently expects batch_size=1")

        block_idx = int(trace_context["block_idx"])
        step_idx = int(trace_context["step_idx"])
        key = (block_idx, int(layer_idx))

        num_total_tokens = int(ln_hidden.shape[1])
        prev_ln_hidden = self.prev_ln_hidden.get(key)
        prev_layer_output = self.prev_layer_output.get(key)

        # Fall back to the normal forward path if skipping is disabled or no previous state exists yet.
        if (not self._should_skip(trace_context)) or prev_ln_hidden is None or prev_layer_output is None:
            return {
                "block_idx": block_idx,
                "step_idx": step_idx,
                "key": key,
                "num_total_tokens": num_total_tokens,
                "num_active_tokens": num_total_tokens,
                "token_cosine": None,
                "layer_similarity": None,
                "prev_ln_hidden": prev_ln_hidden,
                "prev_layer_output": prev_layer_output,
                "did_skip": False,
            }

        prev_ln_hidden = prev_ln_hidden.to(device=ln_hidden.device, dtype=ln_hidden.dtype)
        prev_layer_output = prev_layer_output.to(device=ln_hidden.device, dtype=ln_hidden.dtype)

        # Compare the current layer input against the previous denoising step for this same layer.
        decision = self.policy.build_decision(ln_hidden, prev_ln_hidden)
        layer_similarity = decision["layer_similarity"]
        should_reuse_layer = bool(decision["skip_layer"].item())

        # Keep the final decision aligned with the configured threshold rule.
        expected_skip = bool(layer_similarity.item() >= self.config.threshold)
        if expected_skip != should_reuse_layer:
            raise ValueError("Layer skip decision does not match the configured threshold rule")

        return {
            "block_idx": block_idx,
            "step_idx": step_idx,
            "key": key,
            "num_total_tokens": num_total_tokens,
            "num_active_tokens": 0 if should_reuse_layer else num_total_tokens,
            "token_cosine": decision["token_cosine"],
            "layer_similarity": layer_similarity,
            "prev_ln_hidden": prev_ln_hidden,
            "prev_layer_output": prev_layer_output,
            "did_skip": should_reuse_layer,
        }

    def finish_layer(self, layer_idx, ln_hidden, layer_output, trace_context, skip_plan):
        # Store the current layer state for the next denoising step and write one log record.
        key = skip_plan["key"]

        self.prev_ln_hidden[key] = ln_hidden.detach().cpu()
        self.prev_layer_output[key] = layer_output.detach().cpu()

        if self.stats_recorder is not None:
            self.stats_recorder.record_layer_step(
                block_idx=skip_plan["block_idx"],
                step_idx=skip_plan["step_idx"],
                layer_idx=layer_idx,
                num_total_tokens=skip_plan["num_total_tokens"],
                layer_similarity=skip_plan["layer_similarity"],
                token_cosine=skip_plan["token_cosine"],
                skipped=skip_plan["did_skip"],
                threshold=self.config.threshold,
                aggregation=self.config.aggregation,
            )

        return layer_output
