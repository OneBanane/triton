"""
Embedding layers for the Two-Tower model.

Turns raw user/item features (categorical ids, multi-hot sets, numeric
features, etc.) into dense vectors that feed into the user/item towers.
"""

import torch
from torch import nn


class FeatureEmbedder(nn.Module):
    """Embeds a single categorical feature (e.g. user_id, item_id, category_id)."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # TODO: nn.Embedding(num_embeddings, embedding_dim)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return embedding lookup
        raise NotImplementedError


class MultiFeatureEmbedder(nn.Module):
    """Embeds and concatenates several categorical/numeric features into one vector."""

    def __init__(self, feature_configs: dict):
        super().__init__()
        # TODO: build a ModuleDict of FeatureEmbedder per feature in feature_configs
        raise NotImplementedError

    def forward(self, features: dict) -> torch.Tensor:
        # TODO: embed each feature and concatenate along the last dim
        raise NotImplementedError
