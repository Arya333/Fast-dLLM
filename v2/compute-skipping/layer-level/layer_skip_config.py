from dataclasses import dataclass


@dataclass
class LayerSkipConfig:
    enabled: bool = False
    aggregation: str = "avg"
    threshold: float = 0.99

    def validate(self):
        if self.aggregation not in {"avg", "max"}:
            raise ValueError("aggregation must be 'avg' or 'max'")

        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be between 0 and 1")

    def setting_name(self):
        if not self.enabled:
            return "baseline"
        return f"layer_{self.aggregation}_{self.threshold}"
