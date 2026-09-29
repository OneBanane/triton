"""
CLI entry point for training the Two-Tower recommender on MovieLens.

Loads preprocessed interaction parquet files (see `notebooks/main.ipynb`),
builds a `TwoTowerModel` sized to the resulting user/item vocabularies,
and runs `TwoTowerLightningModule` through a `lightning.pytorch.Trainer`.

Config is composed by Hydra from `configs/config.yaml` and its `model`,
`optimizer`, `dataset` and `training` groups (see `configs/`). Override any
field from the CLI, e.g.:
    uv run python -m triton.training.main
    uv run python -m triton.training.main training.max_epochs=20 training.batch_size=512
    uv run python -m triton.training.main model=two_tower optimizer.learning_rate=1e-4
"""

from __future__ import annotations

import torch
import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import DictConfig
from torch.utils.data import DataLoader, default_collate
from pathlib import Path

from triton.models.recommender.network import TwoTowerModel
from triton.training.dataset import MovieLensDataset
from triton.training.train import TwoTowerLightningModule


def _collate_batch(samples: list[dict]) -> dict:
    """Batches `MovieLensDataset` samples and renames "label" -> "labels" to
    match the batch contract `TwoTowerLightningModule` expects."""
    batch = default_collate(samples)
    batch["labels"] = batch.pop("label")
    return batch


def build_dataloaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader, MovieLensDataset]:
    train_dataset = MovieLensDataset.from_parquet(
        cfg.dataset.train_path, rating_threshold=cfg.training.rating_threshold
    )
    val_dataset = MovieLensDataset.from_parquet(
        cfg.dataset.test_path,
        user_vocab=train_dataset.user_vocab,
        item_vocab=train_dataset.item_vocab,
        genre_vocab=train_dataset.genre_vocab,
        max_genres=train_dataset.max_genres,
        rating_threshold=cfg.training.rating_threshold,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=_collate_batch,
    )

    return train_loader, val_loader, train_dataset


def build_lightning_module(
    cfg: DictConfig, train_dataset: MovieLensDataset
) -> TwoTowerLightningModule:
    model = TwoTowerModel(
        user_feature_configs={
            "user_id": (train_dataset.num_users, cfg.model.id_embedding_dim)
        },
        item_feature_configs={
            "movie_id": (train_dataset.num_items, cfg.model.id_embedding_dim),
            "genres": (
                train_dataset.num_genres + 1,
                cfg.model.genre_embedding_dim,
                "multi_hot",
            ),
        },
        embedding_dim=cfg.model.embedding_dim,
        tower_hidden_dims=list(cfg.model.hidden_dims),
        tower_dropout=cfg.model.dropout,
        tower_normalize=cfg.model.normalize,
        similarity_method=cfg.model.similarity_method,
    )
    return TwoTowerLightningModule(
        model=model, learning_rate=cfg.optimizer.learning_rate
    )


def _build_dummy_input(
    model: TwoTowerModel, batch_size: int = 1
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    def _dummy_features(feature_configs: dict) -> dict[str, torch.Tensor]:
        return {
            name: torch.randint(0, num_embeddings, (batch_size,))
            for name, (num_embeddings, _) in feature_configs.items()
        }

    return (
        _dummy_features(model.user_feature_configs),
        _dummy_features(model.item_feature_configs),
    )


def save_model_to_onnx(cfg: DictConfig, model: TwoTowerModel) -> None:
    dummy_input = _build_dummy_input(model, cfg.training.batch_size)
    output_path = (
        Path(cfg.training.artifacts) / str(cfg.training.model_version) / "model.onnx"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model=model,
            args=(*dummy_input, {}),
            f=output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["user_id", "movie_id"],
            output_names=["output"],
            dynamic_axes={
                "user_id": {0: "batch_size"},
                "movie_id": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

    print("Model was saved in artifacts")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.training.seed)

    train_loader, val_loader, train_dataset = build_dataloaders(cfg)
    lightning_module = build_lightning_module(cfg, train_dataset)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
    )

    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=3)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(
        lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    save_model_to_onnx(cfg, lightning_module.model)


if __name__ == "__main__":
    main()
