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
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedder = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedder(x)


class MultiFeatureEmbedder(nn.Module):
    """Embeds and concatenates several categorical/numeric features into one vector."""

    def __init__(self, feature_configs: dict):
        super().__init__()
        self.feature_configs = feature_configs
        self.feature_embedders_mapping: nn.ModuleDict["str", FeatureEmbedder] = (
            nn.ModuleDict()
        )

        for feature_name, feature_config in self.feature_configs.items():
            self.feature_embedders_mapping[feature_name] = FeatureEmbedder(
                feature_config[0], feature_config[1]
            )

    def forward(self, features: dict) -> torch.Tensor:
        if set(features.keys()) != set(self.feature_embedders_mapping.keys()):
            raise KeyError("`features` keys and `feature_configs` keys should be same")

        res = []
        for feature_name, tensor in features.items():
            embedder = self.feature_embedders_mapping[feature_name]
            res.append(embedder(tensor))
        return torch.cat(res, dim=-1)
