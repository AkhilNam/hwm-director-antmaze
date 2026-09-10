"""Level-2 action posterior: 10-step primitive chunk → latent z.

Original: ``PosteriorContinuous`` in ``pldm/models/misc.py``, wired by
``SequencePredictor`` with ``posterior_input_type=actions``.

Executed MLP (``posterior_arch='32-32'``, ``MLP`` default ``norm=None``,
``activation=relu``):

    flatten [B, 10, 2] → [B, 20]
    Linear 20 → 32 + ReLU
    Linear 32 → 32 + ReLU
    Linear 32 → 16
    chunk → (mu, std_raw) each [B, 8]
    mu  = LayerNorm(mu)
    std = softplus(std_raw) + z_min_std     # z_min_std = 0.05

``z_stochastic = false`` and the executed ``forward_multiple`` both set

    z = mu

``PosteriorContinuous.sample()`` exists but is **not called** on this path.
``std`` is still computed (and stored as ``posterior_vars``) even though it
does not affect ``z``.

``z`` is a **macro-action**, not an (x, y) subgoal, not a state, and not a
policy output. At planning time L2 MPPI proposes ``z`` directly; the
posterior is train-time only.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from hwm_faithful.level2.temporal_abstraction import flatten_action_chunk
from hwm_faithful.shapes import (
    ACTION_DIM,
    L2_STEP_SKIP,
    POSTERIOR_INPUT_DIM,
    Z_DIM,
)


class PosteriorOutput(NamedTuple):
    z: torch.Tensor  # [B, 8] = LayerNorm(mu); the executed latent
    mu: torch.Tensor  # same as z
    std: torch.Tensor  # softplus(raw) + min_std; unused for sampling here


class Level2ActionPosterior(nn.Module):
    """Train-time posterior ``q(z | a_{t:t+9})``."""

    def __init__(
        self,
        input_dim: int = POSTERIOR_INPUT_DIM,
        hidden_arch: str = "32-32",
        z_dim: int = Z_DIM,
        min_std: float = 0.05,
        stochastic: bool = False,
    ) -> None:
        super().__init__()
        if input_dim < 1 or z_dim < 1:
            raise ValueError("input_dim and z_dim must be >= 1")
        self.input_dim = int(input_dim)
        self.z_dim = int(z_dim)
        self.min_std = float(min_std)
        self.stochastic = bool(stochastic)
        hidden = [int(x) for x in hidden_arch.split("-") if x]
        dims = [self.input_dim, *hidden, 2 * self.z_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.posterior_net = nn.Sequential(*layers)
        self.mu_ln = nn.LayerNorm(self.z_dim)

    def forward(self, action_chunk: torch.Tensor) -> PosteriorOutput:
        flat = _as_flat_chunk(action_chunk, self.input_dim)
        mu_raw, std_raw = self.posterior_net(flat).chunk(2, dim=-1)
        mu = self.mu_ln(mu_raw)
        std = F.softplus(std_raw) + self.min_std
        if self.stochastic:
            z = mu + std * torch.randn_like(mu)
        else:
            z = mu
        return PosteriorOutput(z=z, mu=mu, std=std)

    def encode_sequence(self, action_chunks: torch.Tensor) -> PosteriorOutput:
        """``action_chunks`` ``[T, B, 10, 2]`` or ``[T, B, 20]`` → stacked ``[T, B, 8]``."""
        if action_chunks.ndim == 4:
            t, b = action_chunks.shape[:2]
            flat = flatten_action_chunk(action_chunks.reshape(t * b, *action_chunks.shape[2:]))
        elif action_chunks.ndim == 3:
            t, b, d = action_chunks.shape
            if d != self.input_dim:
                raise ValueError(
                    f"flat chunks must be [T, B, {self.input_dim}], got {tuple(action_chunks.shape)}"
                )
            flat = action_chunks.reshape(t * b, d)
        else:
            raise ValueError(
                f"encode_sequence expects [T,B,10,2] or [T,B,20], got {tuple(action_chunks.shape)}"
            )
        out = self.forward(flat)
        return PosteriorOutput(
            z=out.z.view(t, b, self.z_dim),
            mu=out.mu.view(t, b, self.z_dim),
            std=out.std.view(t, b, self.z_dim),
        )


def _as_flat_chunk(action_chunk: torch.Tensor, input_dim: int) -> torch.Tensor:
    if action_chunk.ndim == 2:
        if action_chunk.shape[-1] != input_dim:
            raise ValueError(
                f"flat chunk must be [B, {input_dim}], got {tuple(action_chunk.shape)}"
            )
        return action_chunk
    if action_chunk.ndim == 3:
        if action_chunk.shape[-2:] != (L2_STEP_SKIP, ACTION_DIM) and not (
            action_chunk.shape[-1] == ACTION_DIM
            and action_chunk.shape[-2] * ACTION_DIM == input_dim
        ):
            # still allow [B, skip, 2] for any skip if product matches
            if action_chunk.shape[-1] != ACTION_DIM:
                raise ValueError(
                    f"chunk must be [B, skip, {ACTION_DIM}], got {tuple(action_chunk.shape)}"
                )
        flat = flatten_action_chunk(action_chunk)
        if flat.shape[-1] != input_dim:
            raise ValueError(
                f"flattened chunk dim {flat.shape[-1]} != posterior input {input_dim}"
            )
        return flat
    raise ValueError(
        f"action_chunk must be [B, {input_dim}] or [B, skip, 2], got {tuple(action_chunk.shape)}"
    )
