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
