"""
PyTorch Lightning training loop for the Two-Tower recommender.

Wraps `TwoTowerModel` in a `LightningModule`: turns a batch of
(user_features, item_features, labels) into a BCE-with-logits loss over
the model's similarity scores, and exposes the train/val steps + optimizer
that a `lightning.pytorch.Trainer` drives.

Expected batch contract: a dict with
    - "user_features": dict[str, Tensor]  (fed to `model.encode_user`)
    - "item_features": dict[str, Tensor]  (fed to `model.encode_item`)
    - "labels": Tensor of shape (batch_size,), 0/1 float relevance labels
"""

import lightning.pytorch as pl
import torch
from torch import nn

from triton.models.recommender.network import TwoTowerModel


class TwoTowerLightningModule(pl.LightningModule):
    """Training loop for `TwoTowerModel`: BCE-with-logits loss over similarity scores."""

    def __init__(self, model: TwoTowerModel, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.save_hyperparameters(ignore=["model"])

    def forward(self, user_features: dict, item_features: dict) -> torch.Tensor:
        return self.model(user_features, item_features)

    def _step(self, batch: dict) -> torch.Tensor:
        predict = self(batch["user_features"], batch["item_features"])
        return self.loss_fn(predict, batch["labels"])

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
