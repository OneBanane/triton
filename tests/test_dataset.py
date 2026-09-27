"""Tests for triton.training.dataset.MovieLensDataset.

These tests describe the expected Dataset contract before the class is
implemented (TDD scaffold). They should fail against the current
`NotImplementedError` stubs and pass once `__init__`, `from_parquet`,
`user_vocab`/`item_vocab`, `num_users`/`num_items`, `__len__` and
`__getitem__` are implemented.
"""

import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from triton.models.recommender.network import TwoTowerModel
from triton.training.dataset import MovieLensDataset

RATING_THRESHOLD = 3.5


@pytest.fixture
def raw_interactions() -> pl.DataFrame:
    # 3 unique users (10, 20, 30), 3 unique movies (100, 200, 300).
    return pl.DataFrame(
        {
            "userId": [10, 10, 20, 20, 30],
            "movieId": [100, 200, 100, 300, 200],
            "rating": [4.0, 2.0, 5.0, 1.0, RATING_THRESHOLD],
            "timestamp": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def dataset(raw_interactions) -> MovieLensDataset:
    return MovieLensDataset(raw_interactions, rating_threshold=RATING_THRESHOLD)


class TestMovieLensDataset:
    def test_len_matches_number_of_rows(self, dataset, raw_interactions):
        assert len(dataset) == raw_interactions.height

    def test_getitem_returns_expected_keys(self, dataset):
        sample = dataset[0]
        assert set(sample.keys()) == {"user_features", "item_features", "label"}

    def test_user_features_has_user_id_key(self, dataset):
        sample = dataset[0]
        assert set(sample["user_features"].keys()) == {"user_id"}

    def test_item_features_has_movie_id_key(self, dataset):
        sample = dataset[0]
        assert set(sample["item_features"].keys()) == {"movie_id"}

    def test_user_id_is_scalar_long_tensor(self, dataset):
        user_id = dataset[0]["user_features"]["user_id"]
        assert isinstance(user_id, torch.Tensor)
        assert user_id.dim() == 0
        assert user_id.dtype == torch.long

    def test_movie_id_is_scalar_long_tensor(self, dataset):
        movie_id = dataset[0]["item_features"]["movie_id"]
        assert isinstance(movie_id, torch.Tensor)
        assert movie_id.dim() == 0
        assert movie_id.dtype == torch.long

    def test_label_is_scalar_float_tensor(self, dataset):
        label = dataset[0]["label"]
        assert isinstance(label, torch.Tensor)
        assert label.dim() == 0
        assert label.dtype == torch.float32

    def test_label_above_threshold_is_positive(self, dataset):
        # row 0: rating=4.0 >= 3.5
        assert dataset[0]["label"].item() == pytest.approx(1.0)

    def test_label_below_threshold_is_negative(self, dataset):
        # row 1: rating=2.0 < 3.5
        assert dataset[1]["label"].item() == pytest.approx(0.0)

    def test_label_at_threshold_is_positive(self, dataset):
        # row 4: rating==3.5, inclusive threshold
        assert dataset[4]["label"].item() == pytest.approx(1.0)

    def test_num_users_counts_unique_users(self, dataset):
        assert dataset.num_users == 3

    def test_num_items_counts_unique_movies(self, dataset):
        assert dataset.num_items == 3

    def test_user_ids_remapped_to_contiguous_range(self, dataset):
        for i in range(len(dataset)):
            user_id = dataset[i]["user_features"]["user_id"].item()
            assert 0 <= user_id < dataset.num_users

    def test_movie_ids_remapped_to_contiguous_range(self, dataset):
        for i in range(len(dataset)):
            movie_id = dataset[i]["item_features"]["movie_id"].item()
            assert 0 <= movie_id < dataset.num_items

    def test_same_raw_id_maps_to_same_index(self, dataset, raw_interactions):
        # rows 0 and 1 are both userId=10.
        user_id_0 = dataset[0]["user_features"]["user_id"].item()
        user_id_1 = dataset[1]["user_features"]["user_id"].item()
        assert user_id_0 == user_id_1

    def test_provided_vocab_is_reused_instead_of_rebuilt(
        self, dataset, raw_interactions
    ):
        val_interactions = raw_interactions.head(
            1
        )  # just the first row (userId=10, movieId=100)
        val_dataset = MovieLensDataset(
            val_interactions,
            user_vocab=dataset.user_vocab,
            item_vocab=dataset.item_vocab,
            rating_threshold=RATING_THRESHOLD,
        )

        assert val_dataset.user_vocab == dataset.user_vocab
        assert val_dataset.item_vocab == dataset.item_vocab
        assert val_dataset.num_users == dataset.num_users
        assert val_dataset.num_items == dataset.num_items

        train_user_id = dataset[0]["user_features"]["user_id"].item()
        val_user_id = val_dataset[0]["user_features"]["user_id"].item()
        assert train_user_id == val_user_id

    def test_unknown_id_in_provided_vocab_raises_key_error(self, raw_interactions):
        incomplete_item_vocab = {100: 0, 200: 1}  # missing movieId=300 (row 3)

        with pytest.raises(KeyError):
            MovieLensDataset(raw_interactions, item_vocab=incomplete_item_vocab)

    def test_from_parquet_loads_dataset(self, raw_interactions, tmp_path):
        parquet_path = tmp_path / "interactions.parquet"
        raw_interactions.write_parquet(parquet_path)

        loaded = MovieLensDataset.from_parquet(
            parquet_path, rating_threshold=RATING_THRESHOLD
        )

        assert len(loaded) == raw_interactions.height
        assert loaded.num_users == 3
        assert loaded.num_items == 3

    def test_dataloader_batch_feeds_two_tower_model(self, dataset, batch_size):
        loader = DataLoader(dataset, batch_size=batch_size)
        batch = next(iter(loader))

        model = TwoTowerModel(
            user_feature_configs={"user_id": (dataset.num_users, 8)},
            item_feature_configs={"movie_id": (dataset.num_items, 8)},
            embedding_dim=4,
            tower_hidden_dims=[16],
        )

        scores = model(batch["user_features"], batch["item_features"])
        assert scores.shape == (batch["label"].shape[0],)
