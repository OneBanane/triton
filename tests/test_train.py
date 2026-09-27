"""Tests for triton.training.train.TwoTowerLightningModule.

These tests describe the expected training-loop contract before the
LightningModule is implemented (TDD scaffold). They should fail against
the current `NotImplementedError` stubs and pass once `training_step`,
`validation_step`, `forward`, and `configure_optimizers` are implemented.
"""

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from triton.models.recommender.network import TwoTowerModel
from triton.training.train import TwoTowerLightningModule


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
def model(user_feature_configs, item_feature_configs, embedding_dim) -> TwoTowerModel:
    return TwoTowerModel(
        user_feature_configs=user_feature_configs,
        item_feature_configs=item_feature_configs,
        embedding_dim=embedding_dim,
        tower_hidden_dims=[32, 16],
    )


@pytest.fixture
def lightning_module(model) -> TwoTowerLightningModule:
    return TwoTowerLightningModule(model=model, learning_rate=1e-3)


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


@pytest.fixture
def labels(batch_size) -> torch.Tensor:
    return torch.randint(0, 2, (batch_size,)).float()


@pytest.fixture
def batch(user_batch, item_batch, labels) -> dict:
    return {
        "user_features": user_batch,
        "item_features": item_batch,
        "labels": labels,
    }


class TestTwoTowerLightningModule:
    def test_is_lightning_module(self, lightning_module):
        assert isinstance(lightning_module, pl.LightningModule)

    def test_forward_returns_score_per_pair(
        self, lightning_module, user_batch, item_batch, batch_size
    ):
        out = lightning_module(user_batch, item_batch)
        assert out.shape == (batch_size,)

    def test_training_step_returns_scalar_loss(self, lightning_module, batch):
        loss = lightning_module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0

    def test_training_step_loss_is_finite(self, lightning_module, batch):
        loss = lightning_module.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)

    def test_training_step_loss_requires_grad(self, lightning_module, batch):
        loss = lightning_module.training_step(batch, batch_idx=0)
        assert loss.requires_grad

    def test_validation_step_returns_scalar_loss(self, lightning_module, batch):
        loss = lightning_module.validation_step(batch, batch_idx=0)
        assert loss.dim() == 0

    def test_configure_optimizers_returns_optimizer(self, lightning_module):
        optimizer = lightning_module.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Optimizer)

    def test_configure_optimizers_uses_configured_learning_rate(self, model):
        module = TwoTowerLightningModule(model=model, learning_rate=1e-2)
        optimizer = module.configure_optimizers()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-2)

    def test_gradients_flow_to_model_after_backward(self, lightning_module, batch):
        loss = lightning_module.training_step(batch, batch_idx=0)
        loss.backward()

        grads = [p.grad for p in lightning_module.model.parameters()]
        assert any(g is not None and torch.any(g != 0) for g in grads)

    def test_fast_dev_run_trains_without_error(self, lightning_module, batch):
        train_loader = DataLoader([batch], batch_size=None)
        trainer = pl.Trainer(
            fast_dev_run=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(lightning_module, train_dataloaders=train_loader)
