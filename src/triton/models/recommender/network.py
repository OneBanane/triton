"""
Two-Tower Neural Collaborative Filtering model.

Encodes users and items independently through two separate towers
(see `components/towers.py`), then combines their embeddings with a
similarity function (see `components/similarity.py`) to produce a
match score used for training and top-k retrieval.
"""

import torch
from torch import nn

from .components import MultiFeatureEmbedder, UserTower, ItemTower, SimilarityScorer


class TwoTowerModel(nn.Module):
    """Two-Tower NCF model: user tower + item tower + similarity head."""

    def __init__(
        self,
        user_feature_configs: dict,
        item_feature_configs: dict,
        embedding_dim: int,
        tower_hidden_dims: list[int],
        tower_dropout: float = 0.0,
        tower_normalize: bool = False,
        similarity_method: str = "dot",
    ):
        super().__init__()
        self.user_feature_configs = user_feature_configs
        self.item_feature_configs = item_feature_configs
        self.embedding_dim = embedding_dim
        self.similarity_method = similarity_method

        self.tower_hidden_dims = tower_hidden_dims
        self.tower_dropout = tower_dropout
        self.tower_normalize = tower_normalize

        self.user_input_dim = sum(dim for _, dim in self.user_feature_configs.values())
        self.item_input_dim = sum(dim for _, dim in self.item_feature_configs.values())

        self.user_embedder = MultiFeatureEmbedder(self.user_feature_configs)
        self.user_tower = UserTower(
            input_dim=self.user_input_dim,
            hidden_dims=self.tower_hidden_dims,
            output_dim=self.embedding_dim,
            dropout=self.tower_dropout,
            normalize=self.tower_normalize,
        )

        self.item_embedder = MultiFeatureEmbedder(self.item_feature_configs)
        self.item_tower = ItemTower(
            input_dim=self.item_input_dim,
            hidden_dims=self.tower_hidden_dims,
            output_dim=self.embedding_dim,
            dropout=self.tower_dropout,
            normalize=self.tower_normalize,
        )

        self.similarity_scorer = SimilarityScorer(method=self.similarity_method)

    def encode_user(self, user_features: dict) -> torch.Tensor:
        emb = self.user_embedder(user_features)
        return self.user_tower(emb)

    def encode_item(self, item_features: dict) -> torch.Tensor:
        emb = self.item_embedder(item_features)
        return self.item_tower(emb)

    def forward(self, user_features: dict, item_features: dict) -> torch.Tensor:
        user_emb = self.encode_user(user_features)
        item_emb = self.encode_item(item_features)

        return self.similarity_scorer(user_emb, item_emb)
