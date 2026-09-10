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


def bound_z(
    z: torch.Tensor,
    min_step: float = -2.5,
    max_step: float = 2.5,
    per_dim_min: torch.Tensor | None = None,
    per_dim_max: torch.Tensor | None = None,
    margin: float = 0.1,
) -> torch.Tensor:
    """L2 latent bound as executed when ``latent_actions`` forces clamp.

    Original ``normalize_actions(..., clamp_actions=True)``:

    - If dataset percentile bounds exist: per-dim clamp to
      ``(min_bounds+0.1, max_bounds-0.1)``.
    - Else if last dim is 8 **and** no percentile bounds: a leftover Ant
      joint-limit table (not maze2d). We do **not** use that as the maze
      default; it would prevent faithful Diverse Maze execution.
    - Else: scalar ``torch.clamp(z, min_step, max_step)``.

    M7 default is the scalar YAML ``[-2.5, 2.5]`` per component. Pass
    ``per_dim_min/max`` later for released percentile eval.
    """
    if per_dim_min is not None or per_dim_max is not None:
        if per_dim_min is None or per_dim_max is None:
            raise ValueError("per_dim_min and per_dim_max must be provided together")
        lo = per_dim_min.to(device=z.device, dtype=z.dtype) + margin
        hi = per_dim_max.to(device=z.device, dtype=z.dtype) - margin
        shape = [1] * (z.ndim - 1) + [-1]
        return z.clamp(min=lo.view(*shape), max=hi.view(*shape))
    return z.clamp(min=min_step, max=max_step)


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
