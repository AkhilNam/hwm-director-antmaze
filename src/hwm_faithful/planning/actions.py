"""Action-norm rescaling used by original ``normalize_actions``.

L1 YAML sets ``min_step=0``, ``max_step=1``, ``clamp_actions=False``.
That is **not** a per-component clip to ``[0, 1]``. It rescales the
Euclidean norm of each action vector into ``[min_step, max_step]``.

With ``min_step=0`` this is a no-op for norms already in ``(0, 1]`` and
projects longer vectors onto the unit circle.
"""

from __future__ import annotations

import torch


def rescale_action_norm(
    actions: torch.Tensor,
    min_norm: float = 0.0,
    max_norm: float = 1.0,
    eps: float = 1e-6,
    clamp_components: bool = False,
) -> torch.Tensor:
    """Match original ``normalize_actions`` for non-latent L1 actions."""
    if clamp_components:
        return actions.clamp(min=min_norm, max=max_norm)
    norms = actions.norm(dim=-1, keepdim=True)
    max_norms = torch.ones_like(norms) * max_norm
    min_norms = torch.ones_like(norms) * min_norm
    coeff = torch.min(torch.max(norms, min_norms), max_norms) / (norms + eps)
    return actions * coeff


def unnormalize_action(
    actions: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Original ``Normalizer.unnormalize_action``: ``a * std + mean``.

    M4 synthetic / identity stats use ``mean=None`` (no-op). Dataset maze
    stats are approximately ``mean≈0``, ``std≈0.41`` and are not invented here.
    """
    if mean is None:
        return actions
    if std is None:
        raise ValueError("action std is required when mean is provided")
    mean = mean.to(device=actions.device, dtype=actions.dtype)
    std = std.to(device=actions.device, dtype=actions.dtype)
    return actions * std + mean
