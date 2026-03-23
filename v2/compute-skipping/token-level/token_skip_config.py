# Defines the run-time configuration used by the token-level skipping experiments.
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenSkipConfig:
    enabled: bool = False
    mode: str = "threshold"
    threshold: float = 0.99
    topk_percent: Optional[float] = None

    def validate(self):
        # Keep the public token-skip settings aligned with the two experiment families.
        if self.mode not in {"threshold", "topk"}:
            raise ValueError("mode must be 'threshold' or 'topk'")

        if self.mode == "threshold":
            if not (0.0 <= self.threshold <= 1.0):
                raise ValueError("threshold must be between 0 and 1")

        if self.mode == "topk":
            if self.topk_percent is None:
                raise ValueError("topk_percent must be set for topk mode")
            if not (0.0 < self.topk_percent <= 100.0):
                raise ValueError("topk_percent must be in (0, 100]")

    def setting_name(self):
        # Match each token-skip configuration to the log-folder name used by evaluation and plotting.
        if not self.enabled:
            return "baseline"
        if self.mode == "threshold":
            return f"token_threshold_{self.threshold}"
        return f"token_topk_{int(self.topk_percent)}"
