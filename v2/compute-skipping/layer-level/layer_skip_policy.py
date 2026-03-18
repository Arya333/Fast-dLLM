import torch
import torch.nn.functional as F


class LayerSkipPolicy:
    def __init__(self, config):
        self.config = config
        self.config.validate()

    def token_cosine_similarity(self, current_hidden, prev_hidden):
        current_hidden = current_hidden.float()
        prev_hidden = prev_hidden.float()
        token_cosine = F.cosine_similarity(current_hidden, prev_hidden, dim=-1)
        token_cosine = torch.clamp(token_cosine, min=-1.0, max=1.0)
        return token_cosine

    def aggregate_similarity(self, token_cosine):
        if self.config.aggregation == "avg":
            return token_cosine.mean(dim=-1)
        return token_cosine.max(dim=-1).values

    def build_decision(self, current_hidden, prev_hidden):
        token_cosine = self.token_cosine_similarity(current_hidden, prev_hidden)
        layer_similarity = self.aggregate_similarity(token_cosine)
        skip_layer = layer_similarity >= self.config.threshold

        return {
            "token_cosine": token_cosine,
            "layer_similarity": layer_similarity,
            "skip_layer": skip_layer,
        }
