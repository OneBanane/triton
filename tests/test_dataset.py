"""Tests for triton.training.dataset.MovieLensDataset.

These tests describe the expected Dataset contract before the class is
implemented (TDD scaffold). They should fail against the current
`NotImplementedError` stubs and pass once `__init__`, `from_parquet`,
`user_vocab`/`item_vocab`, `num_users`/`num_items`, `__len__` and
`__getitem__` are implemented.

`TestMovieLensDatasetGenres` further describes the (not yet implemented)
"genres" multi-hot item feature: `raw_interactions` carries a pipe-delimited
`genres` string per row (MovieLens format, e.g. "Action|Comedy"), which
`MovieLensDataset` is expected to:
  - split into tokens and collect into a `genre_vocab: dict[str, int]`
    (built from the split the same way `user_vocab`/`item_vocab` are, and
    reusable across splits via a `genre_vocab` constructor arg);
  - expose `.genre_vocab`, `.num_genres` (== len(genre_vocab)) and
    `.max_genres` (the largest per-movie genre count, also reusable across
    splits via a `max_genres` constructor arg so train/val tensors share a
    shape);
  - return a `"genres"` key in `item_features`: a `(max_genres,)` LongTensor
    of vocab indices, right-padded with the pad index `num_genres` (i.e. one
    past the last real vocab index) for movies with fewer genres.
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
    # movieId is deterministically tied to its genres string:
    #   100 -> "Action|Comedy", 200 -> "Drama", 300 -> "Action|Thriller"
    # so unique genre tokens = {Action, Comedy, Drama, Thriller} (4) and the
    # widest movie (100 or 300) has 2 genres -> max_genres == 2.
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
            item_feature_configs={
                "movie_id": (dataset.num_items, 8),
                "genres": (dataset.num_genres + 1, 4, "multi_hot"),
            },
            embedding_dim=4,
            tower_hidden_dims=[16],
        )

        scores = model(batch["user_features"], batch["item_features"])
        assert scores.shape == (batch["label"].shape[0],)


class TestMovieLensDatasetGenres:
    def test_item_features_has_genres_key(self, dataset):
        sample = dataset[0]
        assert set(sample["item_features"].keys()) == {"movie_id", "genres"}

    def test_genre_vocab_contains_all_unique_genre_tokens(self, dataset):
        assert set(dataset.genre_vocab.keys()) == {
            "Action",
            "Comedy",
            "Drama",
            "Thriller",
        }

    def test_num_genres_counts_unique_genre_tokens(self, dataset):
        assert dataset.num_genres == 4

    def test_max_genres_is_largest_genre_count_for_any_movie(self, dataset):
        assert dataset.max_genres == 2

    def test_genres_tensor_shape_and_dtype(self, dataset):
        genres = dataset[0]["item_features"]["genres"]
        assert isinstance(genres, torch.Tensor)
        assert genres.shape == (dataset.max_genres,)
        assert genres.dtype == torch.long

    def test_genres_tensor_contains_correct_vocab_indices(self, dataset):
        # row 0: movieId=100, genres="Action|Comedy", no padding needed (2 == max_genres)
        genres = dataset[0]["item_features"]["genres"].tolist()
        expected = sorted(dataset.genre_vocab[g] for g in ["Action", "Comedy"])
        assert sorted(genres) == expected

    def test_genres_shorter_than_max_are_padded(self, dataset):
        # row 1: movieId=200, genres="Drama" only -> 1 real id + 1 pad
        genres = dataset[1]["item_features"]["genres"].tolist()
        pad_idx = dataset.num_genres
        assert genres.count(pad_idx) == dataset.max_genres - 1
        assert dataset.genre_vocab["Drama"] in genres

    def test_same_movie_genres_are_consistent_across_rows(self, dataset):
        # rows 0 and 2 are both movieId=100
        genres_0 = dataset[0]["item_features"]["genres"].tolist()
        genres_2 = dataset[2]["item_features"]["genres"].tolist()
        assert genres_0 == genres_2

    def test_provided_genre_vocab_and_max_genres_are_reused(
        self, dataset, raw_interactions
    ):
        val_interactions = raw_interactions.head(1)  # movieId=100, "Action|Comedy"
        val_dataset = MovieLensDataset(
            val_interactions,
            user_vocab=dataset.user_vocab,
            item_vocab=dataset.item_vocab,
            genre_vocab=dataset.genre_vocab,
            max_genres=dataset.max_genres,
            rating_threshold=RATING_THRESHOLD,
        )

        assert val_dataset.genre_vocab == dataset.genre_vocab
        assert val_dataset.num_genres == dataset.num_genres
        assert val_dataset.max_genres == dataset.max_genres

        train_genres = dataset[0]["item_features"]["genres"].tolist()
        val_genres = val_dataset[0]["item_features"]["genres"].tolist()
        assert train_genres == val_genres

    def test_unknown_genre_in_provided_vocab_raises_key_error(self, raw_interactions):
        incomplete_genre_vocab = {
            "Action": 0,
            "Comedy": 1,
            "Drama": 2,
        }  # missing "Thriller"

        with pytest.raises(KeyError):
            MovieLensDataset(raw_interactions, genre_vocab=incomplete_genre_vocab)

    def test_from_parquet_loads_genre_vocab(self, raw_interactions, tmp_path):
        parquet_path = tmp_path / "interactions.parquet"
        raw_interactions.write_parquet(parquet_path)

        loaded = MovieLensDataset.from_parquet(
            parquet_path, rating_threshold=RATING_THRESHOLD
        )

        assert loaded.num_genres == 4
        assert loaded.max_genres == 2

    def test_dataloader_batch_feeds_two_tower_model_with_genres(
        self, dataset, batch_size
    ):
        loader = DataLoader(dataset, batch_size=batch_size)
        batch = next(iter(loader))

        model = TwoTowerModel(
            user_feature_configs={"user_id": (dataset.num_users, 8)},
            item_feature_configs={
                "movie_id": (dataset.num_items, 8),
                "genres": (dataset.num_genres + 1, 4, "multi_hot"),
            },
            embedding_dim=4,
            tower_hidden_dims=[16],
        )

        scores = model(batch["user_features"], batch["item_features"])
        assert scores.shape == (batch["label"].shape[0],)
