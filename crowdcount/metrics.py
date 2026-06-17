from __future__ import annotations

import torch


@torch.no_grad()
def mae(pred_counts: torch.Tensor, true_counts: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error over a batch."""
    pred_counts = pred_counts.detach().float().view(-1)
    true_counts = true_counts.detach().float().view(-1)
    return (pred_counts - true_counts).abs().mean()


@torch.no_grad()
def rmse(pred_counts: torch.Tensor, true_counts: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error over a batch."""
    pred_counts = pred_counts.detach().float().view(-1)
    true_counts = true_counts.detach().float().view(-1)
    return ((pred_counts - true_counts) ** 2).mean().sqrt()
