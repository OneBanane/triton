"""Tests for triton.models.recommender.components.similarity."""

import pytest
import torch
from torch import nn

from triton.models.recommender.components.similarity import SimilarityScorer


class TestSimilarityScorer:
    def test_is_module(self):
        scorer = SimilarityScorer()
        assert isinstance(scorer, nn.Module)

    def test_default_method_is_dot(self, batch_size):
        scorer = SimilarityScorer()
        user_emb = torch.randn(batch_size, 8)
        item_emb = torch.randn(batch_size, 8)

        out = scorer(user_emb, item_emb)
        expected = (user_emb * item_emb).sum(dim=-1)

        assert torch.allclose(out, expected, atol=1e-5)

    def test_output_shape(self, batch_size):
        scorer = SimilarityScorer(method="dot")
        user_emb = torch.randn(batch_size, 8)
        item_emb = torch.randn(batch_size, 8)

        out = scorer(user_emb, item_emb)

        assert out.shape == (batch_size,)

    def test_cosine_similarity_matches_reference(self, batch_size):
        scorer = SimilarityScorer(method="cosine")
        user_emb = torch.randn(batch_size, 8)
        item_emb = torch.randn(batch_size, 8)

        out = scorer(user_emb, item_emb)
        expected = nn.functional.cosine_similarity(user_emb, item_emb, dim=-1)

        assert torch.allclose(out, expected, atol=1e-5)

    def test_cosine_similarity_is_bounded(self, batch_size):
        scorer = SimilarityScorer(method="cosine")
        user_emb = torch.randn(batch_size, 8) * 100
        item_emb = torch.randn(batch_size, 8) * 100

        out = scorer(user_emb, item_emb)

        assert torch.all(out <= 1.0 + 1e-5)
        assert torch.all(out >= -1.0 - 1e-5)

    def test_identical_vectors_have_max_cosine_similarity(self):
        scorer = SimilarityScorer(method="cosine")
        emb = torch.randn(1, 8)

        out = scorer(emb, emb)

        assert torch.allclose(out, torch.ones(1), atol=1e-5)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            SimilarityScorer(method="not-a-real-method")

    def test_gradients_flow_through_scorer(self, batch_size):
        scorer = SimilarityScorer(method="dot")
        user_emb = torch.randn(batch_size, 8, requires_grad=True)
        item_emb = torch.randn(batch_size, 8, requires_grad=True)

        out = scorer(user_emb, item_emb)
        out.sum().backward()

        assert user_emb.grad is not None
        assert item_emb.grad is not None
