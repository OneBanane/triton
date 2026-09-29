"""Tests for triton.training.main genres wiring.

These tests describe the expected contract for feeding the "genres"
multi-hot feature (see `TestMovieLensDatasetGenres` in test_dataset.py and
`TestMultiHotFeatureEmbedder`/mixed-config tests in test_embeddings.py)
through `build_dataloaders`/`build_lightning_module` (TDD scaffold). They
build a hand-rolled `cfg` (no real Hydra config files / no real dataset
parquet files) so they stay isolated from `configs/` and `data/`.

They should fail against the current implementation, which only wires
`user_id`/`movie_id` (no genres, no `cfg.model.genre_embedding_dim`), and
pass once:
  - `build_dataloaders` builds `train_dataset`/`val_dataset` with a shared
    `genre_vocab`/`max_genres` (mirroring how `user_vocab`/`item_vocab` are
    already reused between splits);
  - `build_lightning_module` adds a `"genres"` entry to
    `item_feature_configs`, shaped
    `(train_dataset.num_genres + 1, cfg.model.genre_embedding_dim, "multi_hot")`
    (the `+ 1` accounts for the pad index) — which requires a new
    `cfg.model.genre_embedding_dim` field (added here only to the
    test-local `cfg`; wiring it into `configs/model/two_tower.yaml` is part
    of the implementation).
"""

import polars as pl
import pytest
from omegaconf import DictConfig, OmegaConf

from triton.training.main import build_dataloaders, build_lightning_module

RATING_THRESHOLD = 3.5


@pytest.fixture
def train_interactions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "userId": [10, 10, 20, 20, 30],
            "movieId": [100, 200, 100, 300, 200],
            "rating": [4.0, 2.0, 5.0, 1.0, RATING_THRESHOLD],
            "timestamp": [1, 2, 3, 4, 5],
            "genres": [
                "Action|Comedy",
                "Drama",
                "Action|Comedy",
                "Action|Thriller",
                "Drama",
            ],
        }
    )


@pytest.fixture
def test_interactions() -> pl.DataFrame:
    # only ids/genres already seen in `train_interactions` (no cold-start rows)
    return pl.DataFrame(
        {
            "userId": [10, 20],
            "movieId": [100, 300],
            "rating": [5.0, 2.0],
            "timestamp": [6, 7],
            "genres": ["Action|Comedy", "Action|Thriller"],
        }
    )


@pytest.fixture
def cfg(tmp_path, train_interactions, test_interactions) -> DictConfig:
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train_interactions.write_parquet(train_path)
    test_interactions.write_parquet(test_path)

    return OmegaConf.create(
        {
            "dataset": {"train_path": str(train_path), "test_path": str(test_path)},
            "training": {
                "rating_threshold": RATING_THRESHOLD,
                "batch_size": 2,
                "num_workers": 0,
            },
            "model": {
                "id_embedding_dim": 8,
                "genre_embedding_dim": 4,
                "embedding_dim": 4,
                "hidden_dims": [16],
                "dropout": 0.0,
                "normalize": True,
                "similarity_method": "dot",
            },
            "optimizer": {"learning_rate": 1e-3},
        }
    )


class TestBuildDataloadersGenres:
    def test_train_dataset_exposes_genre_vocab(self, cfg):
        _, _, train_dataset = build_dataloaders(cfg)

        assert set(train_dataset.genre_vocab.keys()) == {
            "Action",
            "Comedy",
            "Drama",
            "Thriller",
        }

    def test_val_dataset_reuses_train_genre_vocab_and_max_genres(self, cfg):
        _, val_loader, train_dataset = build_dataloaders(cfg)
        val_dataset = val_loader.dataset

        assert val_dataset.genre_vocab == train_dataset.genre_vocab
        assert val_dataset.max_genres == train_dataset.max_genres

    def test_batches_include_genres_item_feature(self, cfg):
        train_loader, _, _ = build_dataloaders(cfg)

        batch = next(iter(train_loader))

        assert "genres" in batch["item_features"]


class TestBuildLightningModuleGenres:
    def test_item_feature_configs_include_genres(self, cfg):
        _, _, train_dataset = build_dataloaders(cfg)
        lightning_module = build_lightning_module(cfg, train_dataset)

        assert "genres" in lightning_module.model.item_feature_configs
        num_embeddings, embedding_dim, kind = (
            lightning_module.model.item_feature_configs["genres"]
        )
        assert num_embeddings == train_dataset.num_genres + 1
        assert embedding_dim == cfg.model.genre_embedding_dim
        assert kind == "multi_hot"

    def test_end_to_end_forward_pass_with_genres(self, cfg):
        train_loader, _, train_dataset = build_dataloaders(cfg)
        lightning_module = build_lightning_module(cfg, train_dataset)
        batch = next(iter(train_loader))

        assert "genres" in batch["item_features"]

        scores = lightning_module(batch["user_features"], batch["item_features"])

        assert scores.shape == (batch["labels"].shape[0],)
