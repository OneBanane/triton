"""
Tower modules for the Two-Tower model.

Each tower is an MLP that projects an input embedding (from
`embeddings.py`) into a shared latent space. The user tower and the item
tower share the same architecture but are trained with separate weights.
"""

import torch
import torch.nn.functional as F
from torch import nn


class Tower(nn.Module):
    """Generic MLP tower: stack of Linear -> BatchNorm -> ReLU -> Dropout blocks,
    ending with a Linear(*, output_dim) projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.0,
        normalize: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.normalize = normalize

        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
            layers.append(nn.BatchNorm1d(num_features=hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_features=in_dim, out_features=self.output_dim))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.sequential(x)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


class UserTower(Tower):
    """Tower that encodes user-side features into the shared embedding space."""

    pass


class ItemTower(Tower):
    """Tower that encodes item-side features into the shared embedding space."""

    pass
