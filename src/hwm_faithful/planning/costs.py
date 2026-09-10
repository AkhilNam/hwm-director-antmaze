"""Level-1 representation cost as executed by MPPI ``RunningCost``.

Original MPPI does **not** call ``ReprTargetMPCObjective.__call__``.
``RunningCost`` computes, at each predicted step:

    mean_D (flatten(obs) - flatten(target))^2

and ``mppi_torch._compute_rollout_costs`` **sums** those scalars over
selected timesteps (after the first dynamics step, so ``h0`` is excluded).

``sum_all_diffs=true`` → every predicted step ``t=1..T``.
Otherwise only the last ``sum_last_n`` predicted steps (default 3).

``loss_coeff_first`` / ``loss_coeff_last`` exist on ``ReprTargetMPCObjective``
but are unused in the MPPI running-cost path. Optional ramp is off by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hwm_faithful.shapes import FEATURE_SIZE, VISUAL_CHANNELS


def flatten_spatial(x: torch.Tensor) -> torch.Tensor:
    """Match ``flatten_conv_output`` for 4D ``[N, C, H, W]`` or 5D ``[T, N, C, H, W]``."""
    if x.ndim == 4:
        return x.reshape(x.shape[0], -1)
    if x.ndim == 5:
        t, n = x.shape[:2]
        return x.reshape(t, n, -1)
    raise ValueError(f"expected 4D or 5D spatial tensor, got {tuple(x.shape)}")


def step_mean_sq_error(obs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample mean MSE over feature dims. Returns ``[N]``."""
    obs_f = flatten_spatial(obs) if obs.ndim >= 4 else obs
    tgt_f = flatten_spatial(target) if target.ndim >= 4 else target
    if tgt_f.ndim == 1:
        tgt_f = tgt_f.unsqueeze(0)
    if tgt_f.shape[0] == 1 and obs_f.shape[0] != 1:
        tgt_f = tgt_f.expand(obs_f.shape[0], -1)
    if obs_f.shape != tgt_f.shape:
        raise ValueError(
            f"obs/target feature mismatch: {tuple(obs_f.shape)} vs {tuple(tgt_f.shape)}"
        )
    return (obs_f - tgt_f).pow(2).mean(dim=-1)


def trajectory_repr_cost(
    obs_traj: torch.Tensor,
    target: torch.Tensor,
    *,
    sum_all_diffs: bool = True,
    sum_last_n: int = 3,
    loss_coeff_first: float = 1.0,
    loss_coeff_last: float = 1.0,
    apply_loss_coeff_ramp: bool = False,
) -> torch.Tensor:
    """Cost of a predicted visual trajectory vs a goal map.

    ``obs_traj``: ``[T+1, K, C, H, W]`` including time-0. Returns ``[K]``.
    """
    if obs_traj.ndim != 5:
        raise ValueError(
            f"obs_traj must be [T+1, K, C, H, W], got {tuple(obs_traj.shape)}"
        )
    preds = obs_traj[1:]
    t_pred, k = preds.shape[:2]
    if t_pred < 1:
        raise ValueError("need at least one predicted step")
    if target.ndim == 3:
        target = target.unsqueeze(0)
    flat_pred = flatten_spatial(preds)
    flat_tgt = flatten_spatial(target) if target.ndim >= 4 else target
    if flat_tgt.ndim == 1:
        flat_tgt = flat_tgt.unsqueeze(0)
    if flat_tgt.shape[0] == 1:
        flat_tgt = flat_tgt.expand(k, -1)
    if flat_tgt.shape[0] != k:
        raise ValueError(
            f"target batch {flat_tgt.shape[0]} != K={k}"
        )
    costs = (flat_pred - flat_tgt.unsqueeze(0)).pow(2).mean(dim=-1)  # [T, K]
    if apply_loss_coeff_ramp:
        coeffs = torch.linspace(
            loss_coeff_first,
            loss_coeff_last,
            t_pred,
            dtype=costs.dtype,
            device=costs.device,
        ).view(-1, 1)
        costs = costs * coeffs
    if sum_all_diffs:
        return costs.sum(dim=0)
    n = min(sum_last_n, t_pred)
    return costs[-n:].sum(dim=0)


@dataclass
class RepresentationCost:
    """MPPI ``RunningCost`` for ``cost_entity=obs_component``.

    Operates on visual maps only (16 channels), not fused 18-channel states.
    """

    sum_all_diffs: bool = True
    sum_last_n: int = 3
    loss_coeff_first: float = 1.0
    loss_coeff_last: float = 1.0
    apply_loss_coeff_ramp: bool = False

    def __call__(self, obs_traj: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return trajectory_repr_cost(
            obs_traj,
            target,
            sum_all_diffs=self.sum_all_diffs,
            sum_last_n=self.sum_last_n,
            loss_coeff_first=self.loss_coeff_first,
            loss_coeff_last=self.loss_coeff_last,
            apply_loss_coeff_ramp=self.apply_loss_coeff_ramp,
        )


def validate_visual(x: torch.Tensor, name: str = "visual") -> None:
    if x.ndim not in (3, 4):
        raise ValueError(f"{name} must be [C,H,W] or [B,C,H,W], got {tuple(x.shape)}")
    spatial = x.shape[-3:]
    if spatial != (VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE):
        raise ValueError(
            f"{name} must have shape [..., {VISUAL_CHANNELS}, {FEATURE_SIZE}, "
            f"{FEATURE_SIZE}], got {tuple(x.shape)}"
        )


def z_regularization(
    z: torch.Tensor,
    coeff: float = 0.1,
) -> torch.Tensor:
    """Original latent-action NLL toward ``N(0, 1)``, per candidate.

    ``z`` is ``[K, T, 8]`` (bound sampled macros). Original:

        z_reg = -Normal(0, 1).log_prob(actions).mean(dim=(1, 2)) * coeff

    This is **computed** when ``latent_actions`` is true, then **never
    added** to ``cost_total``. Keep ``coeff`` for diagnostics.
    """
    if z.ndim != 3:
        raise ValueError(f"z must be [K, T, nu], got {tuple(z.shape)}")
    prior = torch.distributions.Normal(
        torch.zeros_like(z), torch.ones_like(z)
    )
    return -prior.log_prob(z).mean(dim=(1, 2)) * coeff
