"""
PyTorch Dataset for the Two-Tower recommender's MovieLens interactions.

Wraps a table of raw interactions (userId, movieId, rating, ...) — the
schema produced by `notebooks/main.ipynb` into `data/preprocessed/*.parquet`
— and turns each row into a sample matching the batch contract expected by
`TwoTowerLightningModule` / `TwoTowerModel` (see `triton.training.train`):
    - user_features: dict[str, Tensor]  -> {"user_id": LongTensor scalar}
    - item_features: dict[str, Tensor]  -> {"movie_id": LongTensor scalar}
    - label: Tensor scalar, 0/1 float relevance label derived from `rating`

Raw MovieLens ids are not contiguous, so they must be remapped to a
[0, num_users) / [0, num_items) index space before being fed to
`nn.Embedding`. When building a dataset for a val/test split, pass in the
`user_vocab`/`item_vocab` built from the train split so indices line up
with the embedding table the model was trained with.
"""

from __future__ import annotations

from pathlib import Path

import torch
import polars as pl
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    """torch Dataset over MovieLens user-item interactions.

    Each sample is a dict with "user_features", "item_features" and
    "label" keys, ready to be batched by a `DataLoader` and fed straight
    into `TwoTowerLightningModule.training_step` / `validation_step`.
    """

    def __init__(
        self,
        interactions: pl.DataFrame,
        user_vocab: dict[int, int] | None = None,
        item_vocab: dict[int, int] | None = None,
        rating_threshold: float = 3.5,
    ):
        self.interactions = interactions
        self.rating_threshold = rating_threshold

        #   if `user_vocab` (resp. `item_vocab`) is None, build one from
        #   this split by mapping each unique raw id in
        #   `interactions["userId"]` (resp. ["movieId"]) to a contiguous
        #   index, e.g. `{raw_id: i for i, raw_id in enumerate(sorted(...))}`.
        #   Otherwise store the given vocab as-is, so a val/test split can
        #   share the train split's embedding index space.
        if user_vocab is None:
            user_vocab = {
                raw_id: i
                for i, raw_id in enumerate(sorted(set(interactions["userId"])))
            }

        if item_vocab is None:
            item_vocab = {
                raw_id: i
                for i, raw_id in enumerate(sorted(set(interactions["movieId"])))
            }

        #   eagerly validate that every raw id in `interactions` is a key
        #   of the (possibly caller-provided) vocab; callers are expected to
        #   have already dropped cold-start rows (see notebooks/main.ipynb),
        #   so a missing id should raise KeyError rather than fail silently.
        #   Vectorized via `is_in` instead of a Python-level loop over the
        #   Series, which is far slower for large interaction tables.
        missing_users = interactions.filter(
            ~pl.col("userId").is_in(list(user_vocab.keys()))
        )["userId"]
        if missing_users.len() > 0:
            raise KeyError(
                f"Every raw id in `interactions` should be a key of the vocab. "
                f"Missing user id: {missing_users[0]}."
            )

        missing_items = interactions.filter(
            ~pl.col("movieId").is_in(list(item_vocab.keys()))
        )["movieId"]
        if missing_items.len() > 0:
            raise KeyError(
                f"Every raw id in `interactions` should be a key of the vocab. "
                f"Missing item id: {missing_items[0]}."
            )

        self._user_vocab = user_vocab
        self._item_vocab = item_vocab

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        user_vocab: dict[int, int] | None = None,
        item_vocab: dict[int, int] | None = None,
        rating_threshold: float = 3.5,
    ) -> MovieLensDataset:
        interactions = pl.read_parquet(path)
        return cls(interactions, user_vocab, item_vocab, rating_threshold)

    @property
    def user_vocab(self) -> dict[int, int]:
        return self._user_vocab

    @property
    def item_vocab(self) -> dict[int, int]:
        return self._item_vocab

    @property
    def num_users(self) -> int:
        return len(self.user_vocab)

    @property
    def num_items(self) -> int:
        return len(self.item_vocab)

    def __len__(self) -> int:
        return self.interactions.height

    def __getitem__(self, index: int) -> dict:
        #   look up row `index`, remap its raw userId/movieId through
        #   `self.user_vocab`/`self.item_vocab`, and return:
        #   {
        #       "user_features": {"user_id": torch.tensor(<mapped user idx>)},
        #       "item_features": {"movie_id": torch.tensor(<mapped item idx>)},
        #       "label": torch.tensor(float(rating >= self.rating_threshold)),
        #   }
        raw_user_id = self.interactions["userId"][index]
        raw_item_id = self.interactions["movieId"][index]
        rating = self.interactions["rating"][index]

        user_id = self.user_vocab[raw_user_id]
        item_id = self.item_vocab[raw_item_id]

        return {
            "user_features": {"user_id": torch.tensor(user_id)},
            "item_features": {"movie_id": torch.tensor(item_id)},
            "label": torch.tensor(float(rating >= self.rating_threshold)),
        }
