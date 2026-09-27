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
        # TODO: store the scoring method ("dot", "cosine", ...)
        raise NotImplementedError

    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        # TODO: compute similarity score between user_emb and item_emb
        raise NotImplementedError
