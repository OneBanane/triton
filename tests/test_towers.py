"""Tests for triton.models.recommender.components.towers."""

import pytest
import torch
from torch import nn

from triton.models.recommender.components.towers import ItemTower, Tower, UserTower


@pytest.fixture(params=[Tower, UserTower, ItemTower])
def tower_cls(request):
    return request.param


class TestTower:
    def test_is_module(self, tower_cls):
        tower = tower_cls(input_dim=16, hidden_dims=[8], output_dim=4)
        assert isinstance(tower, nn.Module)

    def test_output_shape_with_hidden_layers(self, batch_size, tower_cls):
        tower = tower_cls(input_dim=16, hidden_dims=[32, 8], output_dim=4)
        x = torch.randn(batch_size, 16)

        out = tower(x)

        assert out.shape == (batch_size, 4)

    def test_output_shape_with_no_hidden_layers(self, batch_size, tower_cls):
        tower = tower_cls(input_dim=16, hidden_dims=[], output_dim=4)
        x = torch.randn(batch_size, 16)

        out = tower(x)

        assert out.shape == (batch_size, 4)

    def test_output_has_no_nans(self, batch_size, tower_cls):
        tower = tower_cls(input_dim=16, hidden_dims=[8], output_dim=4)
        x = torch.randn(batch_size, 16)

        out = tower(x)

        assert not torch.isnan(out).any()

    def test_gradients_flow_to_parameters(self, batch_size, tower_cls):
        tower = tower_cls(input_dim=16, hidden_dims=[8], output_dim=4)
        x = torch.randn(batch_size, 16)

        out = tower(x)
        out.sum().backward()

        grads = [p.grad for p in tower.parameters()]
        assert len(grads) > 0
        assert any(g is not None and torch.any(g != 0) for g in grads)

    def test_accepts_dropout_argument(self, tower_cls):
        tower = tower_cls(input_dim=16, hidden_dims=[8], output_dim=4, dropout=0.5)
        assert isinstance(tower, nn.Module)


class TestTowerSubclassing:
    def test_user_tower_is_a_tower(self):
        tower = UserTower(input_dim=16, hidden_dims=[8], output_dim=4)
        assert isinstance(tower, Tower)

    def test_item_tower_is_a_tower(self):
        tower = ItemTower(input_dim=16, hidden_dims=[8], output_dim=4)
        assert isinstance(tower, Tower)
