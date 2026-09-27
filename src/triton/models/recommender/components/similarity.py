"""
Similarity/scoring functions for the Two-Tower model.

Combines the user tower output and the item tower output into a single
match score (e.g. dot product, cosine similarity) used for training
(sampled softmax / BPR / BCE) and inference (top-k retrieval).
"""

import torch
from torch import nn


class SimilarityScorer(nn.Module):
    """Scores a batch of (user_embedding, item_embedding) pairs."""

    def __init__(self, method: str = "dot"):
        super().__init__()
        self.method = method

        self._post_init()

    def _post_init(self):
        supported_methods = {"dot", "cosine"}

        if self.method not in supported_methods:
            raise ValueError(
                f"Unsupported method: {self.method}. " f"Available: {supported_methods}"
            )

    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        res = torch.tensor([])
        if self.method == "dot":
            res = torch.sum(user_emb * item_emb, dim=-1)
        elif self.method == "cosine":
            res = torch.cosine_similarity(user_emb, item_emb, dim=-1)
        return res
