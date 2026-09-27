"""Tests for triton.models.recommender.components.embeddings.

Feature configs are expected as {feature_name: (num_embeddings, embedding_dim)}.
"""

import pytest
import torch
from torch import nn

from triton.models.recommender.components.embeddings import (
    FeatureEmbedder,
    MultiFeatureEmbedder,
)


class TestFeatureEmbedder:
    def test_is_module(self):
        embedder = FeatureEmbedder(num_embeddings=10, embedding_dim=4)
        assert isinstance(embedder, nn.Module)

    def test_output_shape(self, batch_size):
        embedder = FeatureEmbedder(num_embeddings=10, embedding_dim=4)
        x = torch.randint(0, 10, (batch_size,))

        out = embedder(x)

        assert out.shape == (batch_size, 4)

    def test_output_is_differentiable(self, batch_size):
        embedder = FeatureEmbedder(num_embeddings=10, embedding_dim=4)
        x = torch.randint(0, 10, (batch_size,))

        out = embedder(x)
        out.sum().backward()

        grads = [p.grad for p in embedder.parameters()]
        assert any(g is not None and torch.any(g != 0) for g in grads)

    def test_different_indices_give_different_embeddings(self):
        embedder = FeatureEmbedder(num_embeddings=10, embedding_dim=4)

        out = embedder(torch.tensor([0, 1]))

        assert not torch.allclose(out[0], out[1])


class TestMultiFeatureEmbedder:
    def test_is_module(self, multi_feature_config):
        embedder = MultiFeatureEmbedder(multi_feature_config)
        assert isinstance(embedder, nn.Module)

    def test_output_concatenates_all_feature_dims(
        self, batch_size, multi_feature_config, multi_feature_batch
    ):
        embedder = MultiFeatureEmbedder(multi_feature_config)
        expected_dim = sum(dim for _, dim in multi_feature_config.values())

        out = embedder(multi_feature_batch)

        assert out.shape == (batch_size, expected_dim)

    def test_single_feature_matches_feature_embedder_dim(
        self, batch_size, single_feature_config
    ):
        embedder = MultiFeatureEmbedder(single_feature_config)
        num_embeddings, embedding_dim = single_feature_config["item_id"]
        x = {"item_id": torch.randint(0, num_embeddings, (batch_size,))}

        out = embedder(x)

        assert out.shape == (batch_size, embedding_dim)

    def test_output_is_differentiable(self, multi_feature_config, multi_feature_batch):
        embedder = MultiFeatureEmbedder(multi_feature_config)

        out = embedder(multi_feature_batch)
        out.sum().backward()

        grads = [p.grad for p in embedder.parameters()]
        assert any(g is not None and torch.any(g != 0) for g in grads)

    def test_missing_feature_raises(self, multi_feature_config, multi_feature_batch):
        embedder = MultiFeatureEmbedder(multi_feature_config)
        incomplete_batch = dict(multi_feature_batch)
        incomplete_batch.pop("user_id")

        with pytest.raises((KeyError, ValueError)):
            embedder(incomplete_batch)
