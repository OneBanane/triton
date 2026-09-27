"""
Tower modules for the Two-Tower model.

Each tower is an MLP that projects an input embedding (from
`embeddings.py`) into a shared latent space. The user tower and the item
tower share the same architecture but are trained with separate weights.
"""

import torch
from torch import nn


class Tower(nn.Module):
    """Generic MLP tower: stack of Linear -> Activation -> (Dropout) blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # TODO: build nn.Sequential of Linear/BatchNorm/ReLU/Dropout layers
        # ending with a Linear(*, output_dim) projection
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: forward pass through the MLP, optionally L2-normalize output
        raise NotImplementedError


class UserTower(Tower):
    """Tower that encodes user-side features into the shared embedding space."""

    pass


class ItemTower(Tower):
    """Tower that encodes item-side features into the shared embedding space."""

    pass
