"""Shared fixtures for recommender component tests."""

import pytest
import torch


@pytest.fixture
def batch_size() -> int:
    return 4


@pytest.fixture
def single_feature_config() -> dict:
    """One categorical feature: (num_embeddings, embedding_dim)."""
    return {"item_id": (100, 8)}


@pytest.fixture
def multi_feature_config() -> dict:
    """Several categorical features with different embedding dims."""
    return {
        "user_id": (1000, 16),
        "country": (50, 4),
        "device": (10, 2),
    }


@pytest.fixture
def multi_feature_batch(batch_size, multi_feature_config) -> dict:
    return {
        name: torch.randint(0, num_embeddings, (batch_size,))
        for name, (num_embeddings, _) in multi_feature_config.items()
    }


@pytest.fixture
def max_multi_hot_len() -> int:
    return 3


@pytest.fixture
def multi_hot_feature_config() -> dict:
    """One multi-hot categorical feature: (num_embeddings, embedding_dim, "multi_hot")."""
    return {"genres": (20, 6, "multi_hot")}


@pytest.fixture
def multi_hot_batch(batch_size, multi_hot_feature_config, max_multi_hot_len) -> dict:
    """Padded (batch, max_len) LongTensor per multi-hot feature."""
    return {
        name: torch.randint(0, num_embeddings, (batch_size, max_multi_hot_len))
        for name, (num_embeddings, _, _) in multi_hot_feature_config.items()
    }


@pytest.fixture
def mixed_feature_config(multi_feature_config, multi_hot_feature_config) -> dict:
    """Combines single-id (2-tuple) and multi-hot (3-tuple) feature configs."""
    return {**multi_feature_config, **multi_hot_feature_config}


@pytest.fixture
def mixed_batch(multi_feature_batch, multi_hot_batch) -> dict:
    return {**multi_feature_batch, **multi_hot_batch}
