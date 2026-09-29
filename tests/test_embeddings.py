"""Tests for triton.models.recommender.components.embeddings.

Single-id feature configs are {feature_name: (num_embeddings, embedding_dim)}.
Multi-hot feature configs (e.g. a movie's set of genres) are
{feature_name: (num_embeddings, embedding_dim, "multi_hot")}: the extra
"multi_hot" tag tells `MultiFeatureEmbedder` to route the feature through a
`MultiHotFeatureEmbedder` (pooling a padded (batch, max_len) LongTensor)
instead of a plain `FeatureEmbedder` (which expects a (batch,) LongTensor of
single ids). `MultiFeatureEmbedder` must accept a mix of both config shapes
in the same `feature_configs` dict.
"""

import pytest
import torch
from torch import nn

from triton.models.recommender.components.embeddings import (
    FeatureEmbedder,
    MultiFeatureEmbedder,
    MultiHotFeatureEmbedder,
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

    def test_mixed_single_and_multi_hot_output_dim(
        self, batch_size, mixed_feature_config, mixed_batch
    ):
        embedder = MultiFeatureEmbedder(mixed_feature_config)
        expected_dim = sum(config[1] for config in mixed_feature_config.values())

        out = embedder(mixed_batch)

        assert out.shape == (batch_size, expected_dim)

    def test_mixed_single_and_multi_hot_is_differentiable(
        self, mixed_feature_config, mixed_batch
    ):
        embedder = MultiFeatureEmbedder(mixed_feature_config)

        out = embedder(mixed_batch)
        out.sum().backward()

        grads = [p.grad for p in embedder.parameters()]
        assert any(g is not None and torch.any(g != 0) for g in grads)

    def test_multi_hot_feature_uses_multi_hot_feature_embedder(
        self, mixed_feature_config
    ):
        embedder = MultiFeatureEmbedder(mixed_feature_config)
        assert isinstance(
            embedder.feature_embedders_mapping["genres"], MultiHotFeatureEmbedder
        )

    def test_single_id_feature_still_uses_feature_embedder(self, mixed_feature_config):
        embedder = MultiFeatureEmbedder(mixed_feature_config)
        assert isinstance(
            embedder.feature_embedders_mapping["user_id"], FeatureEmbedder
        )


class TestMultiHotFeatureEmbedder:
    def test_is_module(self, multi_hot_feature_config):
        num_embeddings, embedding_dim, _ = multi_hot_feature_config["genres"]
        embedder = MultiHotFeatureEmbedder(num_embeddings, embedding_dim)
        assert isinstance(embedder, nn.Module)

    def test_output_shape(
        self, batch_size, multi_hot_feature_config, max_multi_hot_len
    ):
        num_embeddings, embedding_dim, _ = multi_hot_feature_config["genres"]
        embedder = MultiHotFeatureEmbedder(num_embeddings, embedding_dim)
        x = torch.randint(0, num_embeddings, (batch_size, max_multi_hot_len))

        out = embedder(x)

        assert out.shape == (batch_size, embedding_dim)

    def test_output_is_differentiable(
        self, batch_size, multi_hot_feature_config, max_multi_hot_len
    ):
        num_embeddings, embedding_dim, _ = multi_hot_feature_config["genres"]
        embedder = MultiHotFeatureEmbedder(num_embeddings, embedding_dim)
        x = torch.randint(0, num_embeddings, (batch_size, max_multi_hot_len))

        out = embedder(x)
        out.sum().backward()

        grads = [p.grad for p in embedder.parameters()]
        assert any(g is not None and torch.any(g != 0) for g in grads)

    def test_different_id_sets_give_different_embeddings(self):
        embedder = MultiHotFeatureEmbedder(num_embeddings=10, embedding_dim=4)

        out = embedder(torch.tensor([[0, 1], [2, 3]]))

        assert not torch.allclose(out[0], out[1])

    def test_padding_idx_is_excluded_from_pooling(self):
        num_embeddings = 5
        padding_idx = 4
        embedder = MultiHotFeatureEmbedder(
            num_embeddings, embedding_dim=6, padding_idx=padding_idx, mode="mean"
        )

        unpadded = torch.tensor([[0, 1]])
        padded_with_trailing_pad = torch.tensor([[0, 1, padding_idx]])

        out_unpadded = embedder(unpadded)
        out_padded = embedder(padded_with_trailing_pad)

        assert torch.allclose(out_unpadded, out_padded)
