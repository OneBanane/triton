"""
Two-Tower Neural Collaborative Filtering model.

Encodes users and items independently through two separate towers
(see `components/towers.py`), then combines their embeddings with a
similarity function (see `components/similarity.py`) to produce a
match score used for training and top-k retrieval.
"""

import torch
from torch import nn


class TwoTowerModel(nn.Module):
    """Two-Tower NCF model: user tower + item tower + similarity head."""

    def __init__(
        self,
        user_feature_configs: dict,
        item_feature_configs: dict,
        embedding_dim: int,
        tower_hidden_dims: list[int],
    ):
        super().__init__()
        # TODO: build user_embedder (MultiFeatureEmbedder) + user_tower (UserTower)
        # TODO: build item_embedder (MultiFeatureEmbedder) + item_tower (ItemTower)
        # TODO: build similarity scorer (SimilarityScorer)
        raise NotImplementedError

    def encode_user(self, user_features: dict) -> torch.Tensor:
        # TODO: user_embedder -> user_tower
        raise NotImplementedError

    def encode_item(self, item_features: dict) -> torch.Tensor:
        # TODO: item_embedder -> item_tower
        raise NotImplementedError

    def forward(self, user_features: dict, item_features: dict) -> torch.Tensor:
        # TODO: encode both towers and score the pair with the similarity head
        raise NotImplementedError
