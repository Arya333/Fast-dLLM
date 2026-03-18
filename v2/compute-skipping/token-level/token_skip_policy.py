import math

import torch
import torch.nn.functional as F


class TokenSkipPolicy:
    def __init__(self, config):
        self.config = config
        self.config.validate()

    def token_cosine_similarity(self, current_hidden, prev_hidden):
        current_hidden = current_hidden.float()
        prev_hidden = prev_hidden.float()
        token_cosine = F.cosine_similarity(current_hidden, prev_hidden, dim=-1)
        token_cosine = torch.clamp(token_cosine, min=-1.0, max=1.0)
        return token_cosine

    def build_masks(self, current_hidden, prev_hidden):
        token_cosine = self.token_cosine_similarity(current_hidden, prev_hidden)

        if self.config.mode == "threshold":
            # Recompute tokens whose similarity is below the threshold
            active_mask = token_cosine < self.config.threshold
        else:
            batch_size, num_tokens = token_cosine.shape
            k = max(1, math.ceil(num_tokens * self.config.topk_percent / 100.0))

            # The most changed tokens are the ones with the lowest cosine similarity
            changed_score = 1.0 - token_cosine
            topk_indices = torch.topk(changed_score, k=k, dim=-1).indices

            active_mask = torch.zeros_like(token_cosine, dtype=torch.bool)
            active_mask.scatter_(1, topk_indices, True)

        reuse_mask = ~active_mask

        return {
            "token_cosine": token_cosine,
            "active_mask": active_mask,
            "reuse_mask": reuse_mask,
        }
