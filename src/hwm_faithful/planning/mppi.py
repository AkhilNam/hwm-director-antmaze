"""MPPI core matching original ``pldm/planning/planners/mppi_torch.py``.

Williams et al. 2017 as copied into HWM (UM-ARM-Lab / pytorch_mppi):

    ε ~ N(0, Σ),  Σ = noise_sigma · I
    ũ = bound(U + ε)
    ε' = ũ - U
    c_k = rollout_cost(ũ_k) + λ Σ_t U_t · (Σ^{-1} ε'_t)
    β = min_k c_k
    ω_k = exp(-(c_k - β) / λ) / Σ_j exp(-(c_j - β) / λ)
    U ← U + Σ_k ω_k ε'_k

One ``command()`` = one sample-and-update. ``num_refinement_steps`` is stored
in original ``MPPIPlanner`` but never looped. L1 ``z_reg_coeff`` is unused
because ``latent_actions`` is false. L2 computes ``z_reg`` but does not add
it to ``cost_total`` (``apply_z_reg=False``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import torch

from hwm_faithful.planning.actions import rescale_action_norm
from hwm_faithful.shapes import ACTION_DIM


class MPPIStepResult(NamedTuple):
    U: torch.Tensor  # [T, nu]
    weights: torch.Tensor  # [K]
    costs: torch.Tensor  # [K]
    rollout_costs: torch.Tensor  # [K]
    perturbation_costs: torch.Tensor  # [K]
    noise: torch.Tensor  # [K, T, nu] bounded residual
    perturbed_actions: torch.Tensor  # [K, T, nu]


def mppi_weights(costs: torch.Tensor, lambda_: float) -> torch.Tensor:
    """Original ``_ensure_non_zero`` + normalize. ``costs`` is ``[K]``."""
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be > 0, got {lambda_}")
    shifted = torch.exp(-(costs - costs.min()) / lambda_)
    eta = shifted.sum()
    if eta <= 0 or not torch.isfinite(eta):
        raise RuntimeError("MPPI weight normalizer is not finite/positive")
    return shifted / eta


def sample_noise(
    num_samples: int,
    horizon: int,
    action_dim: int,
    noise_sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """``ε ~ N(0, diag(noise_sigma))`` → ``[K, T, nu]``. ``noise_sigma`` is variance."""
    std = float(noise_sigma) ** 0.5
    noise = torch.randn(
        num_samples,
        horizon,
        action_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return noise * std


def bound_plan(
    actions: torch.Tensor,
    min_step: float,
    max_step: float,
    clamp_components: bool = False,
) -> torch.Tensor:
    return rescale_action_norm(
        actions,
        min_norm=min_step,
        max_norm=max_step,
        clamp_components=clamp_components,
    )


class MPPI:
    """Single-environment MPPI controller (original one ``MPPI`` per env)."""

    def __init__(
        self,
        config: Any,
        action_dim: int = ACTION_DIM,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
        horizon: int = 15,
        bound_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.config = config
        self.action_dim = int(action_dim)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.generator = generator
        # Original ``MPPI.__init__`` defaults ``horizon=15``; ``plan()`` then
        # ``change_horizon(plan_size)`` (truncate or pad with zeros).
        self.T = int(horizon)
        self.u_init = torch.zeros(self.action_dim, device=self.device, dtype=self.dtype)
        self.U = self._init_U(self.T)
        self.bound_fn = bound_fn

    def _init_U(self, horizon: int) -> torch.Tensor:
        return sample_noise(
            1,
            horizon,
            self.action_dim,
            self.config.noise_sigma,
            device=self.device,
            dtype=self.dtype,
            generator=self.generator,
        ).squeeze(0)

    def reset(self, horizon: int | None = None) -> None:
        if horizon is not None:
            self.T = int(horizon)
        self.U = self._init_U(self.T)

    def change_horizon(self, horizon: int) -> None:
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if horizon < self.U.shape[0]:
            self.U = self.U[:horizon]
        elif horizon > self.U.shape[0]:
            pad = self.u_init.repeat(horizon - self.U.shape[0], 1)
            self.U = torch.cat([self.U, pad], dim=0)
        self.T = horizon

    def shift_nominal_trajectory(self) -> None:
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init

    def command(
        self,
        rollout_cost_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        shift_nominal_trajectory: bool = False,
    ) -> MPPIStepResult:
        """One MPPI update. ``rollout_cost_fn(actions [K,T,nu]) -> costs [K]``."""
        if shift_nominal_trajectory:
            self.shift_nominal_trajectory()
        cfg = self.config
        noise = sample_noise(
            cfg.num_samples,
            self.T,
            self.action_dim,
            cfg.noise_sigma,
            device=self.U.device,
            dtype=self.U.dtype,
            generator=self.generator,
        )
        perturbed = self._bound(self.U.unsqueeze(0) + noise)
        bounded_noise = perturbed - self.U.unsqueeze(0)
        sigma_inv = 1.0 / cfg.noise_sigma
        action_cost = cfg.lambda_ * bounded_noise * sigma_inv
        perturbation_cost = (self.U.unsqueeze(0) * action_cost).sum(dim=(1, 2))
        rollout_cost = rollout_cost_fn(perturbed)
        if rollout_cost.shape != (cfg.num_samples,):
            raise ValueError(
                f"rollout_cost_fn must return [{cfg.num_samples}], got {tuple(rollout_cost.shape)}"
            )
        costs = rollout_cost + perturbation_cost
        weights = mppi_weights(costs, cfg.lambda_)
        delta = (weights.view(-1, 1, 1) * bounded_noise).sum(dim=0)
        self.U = self.U + delta
        return MPPIStepResult(
            U=self.U,
            weights=weights,
            costs=costs,
            rollout_costs=rollout_cost,
            perturbation_costs=perturbation_cost,
            noise=bounded_noise,
            perturbed_actions=perturbed,
        )

    def _bound(self, actions: torch.Tensor) -> torch.Tensor:
        if self.bound_fn is not None:
            return self.bound_fn(actions)
        cfg = self.config
        return bound_plan(
            actions,
            cfg.min_step,
            cfg.max_step,
            clamp_components=cfg.clamp_actions,
        )
