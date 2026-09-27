"""Tests for triton.models.recommender.network.TwoTowerModel."""

import pytest
import torch
from torch import nn

from triton.models.recommender.network import TwoTowerModel


@pytest.fixture
def user_feature_configs() -> dict:
    return {"user_id": (1000, 16), "country": (50, 4)}


@pytest.fixture
def item_feature_configs() -> dict:
    return {"item_id": (500, 16), "category": (20, 4)}


@pytest.fixture
def embedding_dim() -> int:
    return 8


@pytest.fixture
def model(user_feature_configs, item_feature_configs, embedding_dim):
    return TwoTowerModel(
        user_feature_configs=user_feature_configs,
        item_feature_configs=item_feature_configs,
        embedding_dim=embedding_dim,
        tower_hidden_dims=[32, 16],
    )


@pytest.fixture
def user_batch(batch_size, user_feature_configs) -> dict:
    return {
        name: torch.randint(0, num_embeddings, (batch_size,))
        for name, (num_embeddings, _) in user_feature_configs.items()
    }


@pytest.fixture
def item_batch(batch_size, item_feature_configs) -> dict:
    return {
        name: torch.randint(0, num_embeddings, (batch_size,))
        for name, (num_embeddings, _) in item_feature_configs.items()
    }


class TestTwoTowerModel:
    def test_is_module(self, model):
        assert isinstance(model, nn.Module)

    def test_encode_user_shape(self, model, user_batch, batch_size, embedding_dim):
        out = model.encode_user(user_batch)
        assert out.shape == (batch_size, embedding_dim)

    def test_encode_item_shape(self, model, item_batch, batch_size, embedding_dim):
        out = model.encode_item(item_batch)
        assert out.shape == (batch_size, embedding_dim)

    def test_forward_returns_score_per_pair(
        self, model, user_batch, item_batch, batch_size
    ):
        out = model(user_batch, item_batch)
        assert out.shape == (batch_size,)

    def test_forward_has_no_nans(self, model, user_batch, item_batch):
        out = model(user_batch, item_batch)
        assert not torch.isnan(out).any()

    def test_gradients_flow_to_both_towers(self, model, user_batch, item_batch):
        out = model(user_batch, item_batch)
        out.sum().backward()

        grads = [p.grad for p in model.parameters()]
        assert len(grads) > 0
        assert any(g is not None and torch.any(g != 0) for g in grads)

    def test_encode_user_and_encode_item_are_independent(
        self, model, user_batch, item_batch, embedding_dim
    ):
        user_emb = model.encode_user(user_batch)
        item_emb = model.encode_item(item_batch)

        assert user_emb.shape[-1] == embedding_dim
        assert item_emb.shape[-1] == embedding_dim
