"""
CLI entry point for training the Two-Tower recommender on MovieLens.

Loads preprocessed interaction parquet files (see `notebooks/main.ipynb`),
builds a `TwoTowerModel` sized to the resulting user/item vocabularies,
and runs `TwoTowerLightningModule` through a `lightning.pytorch.Trainer`.

Usage:
    uv run python -m triton.models.training.main
    uv run python -m triton.models.training.main --max-epochs 20 --batch-size 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, default_collate

from triton.models.recommender.network import TwoTowerModel
from triton.training.dataset import MovieLensDataset
from triton.training.train import TwoTowerLightningModule

DEFAULT_TRAIN_PATH = Path("data/preprocessed/train.parquet")
DEFAULT_VAL_PATH = Path("data/preprocessed/test.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Two-Tower recommender.")

    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--rating-threshold", type=float, default=3.5)

    parser.add_argument("--id-embedding-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--tower-hidden-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--tower-dropout", type=float, default=0.0)
    parser.add_argument("--tower-normalize", action="store_true")
    parser.add_argument("--similarity-method", choices=["dot", "cosine"], default="dot")

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/recommender")
    )

    return parser.parse_args()


def _collate_batch(samples: list[dict]) -> dict:
    """Batches `MovieLensDataset` samples and renames "label" -> "labels" to
    match the batch contract `TwoTowerLightningModule` expects."""
    batch = default_collate(samples)
    batch["labels"] = batch.pop("label")
    return batch


def build_dataloaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, MovieLensDataset]:
    train_dataset = MovieLensDataset.from_parquet(
        args.train_path, rating_threshold=args.rating_threshold
    )
    val_dataset = MovieLensDataset.from_parquet(
        args.val_path,
        user_vocab=train_dataset.user_vocab,
        item_vocab=train_dataset.item_vocab,
        rating_threshold=args.rating_threshold,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_batch,
    )

    return train_loader, val_loader, train_dataset


def build_lightning_module(
    args: argparse.Namespace, train_dataset: MovieLensDataset
) -> TwoTowerLightningModule:
    model = TwoTowerModel(
        user_feature_configs={
            "user_id": (train_dataset.num_users, args.id_embedding_dim)
        },
        item_feature_configs={
            "movie_id": (train_dataset.num_items, args.id_embedding_dim)
        },
        embedding_dim=args.embedding_dim,
        tower_hidden_dims=args.tower_hidden_dims,
        tower_dropout=args.tower_dropout,
        tower_normalize=args.tower_normalize,
        similarity_method=args.similarity_method,
    )
    return TwoTowerLightningModule(model=model, learning_rate=args.learning_rate)


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)

    train_loader, val_loader, train_dataset = build_dataloaders(args)
    lightning_module = build_lightning_module(args, train_dataset)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
