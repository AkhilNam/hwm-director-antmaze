"""Held-out L1 prediction and representation diagnostics.

No new learned baselines. No-change representation is the only comparison.
"""

from __future__ import annotations

from typing import Any

import torch

from hwm_faithful.losses.vicreg_obs import cov_offdiag, flatten_spatial, prediction_mse
from hwm_faithful.models.level1_world_model import Level1WorldModel
from hwm_faithful.shapes import FEATURE_SIZE, VISUAL_CHANNELS


HORIZONS = (1, 5, 10, 14)
COLLAPSE_STD_THRESH = 1e-3


@torch.no_grad()
def prediction_diagnostics(
    model: Level1WorldModel,
    images: torch.Tensor,
    proprio: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, Any]:
    """``images/proprio/actions`` are time-leading ``[T, B, ...]``."""
    model.eval()
    encoded, rolled = model.encode_and_rollout(images, proprio, actions)
    obs_gt = encoded.visual
    obs_pred = rolled.obs_component
    prop_gt = encoded.proprio_map
    prop_pred = rolled.proprio_component

    one_step_visual = float(prediction_mse(obs_gt[1:], obs_pred[1:]).item())
    one_step_proprio = float(prediction_mse(prop_gt[1:], prop_pred[1:]).item())
    one_step_visual_nc = float(prediction_mse(obs_gt[1:], obs_gt[:1].expand_as(obs_gt[1:])).item())
    one_step_proprio_nc = float(
        prediction_mse(prop_gt[1:], prop_gt[:1].expand_as(prop_gt[1:])).item()
    )

    multi: dict[str, Any] = {}
    t = obs_gt.shape[0]
    for h in HORIZONS:
        if h >= t:
            continue
        vis = float(prediction_mse(obs_gt[h : h + 1], obs_pred[h : h + 1]).item())
        vis_nc = float(prediction_mse(obs_gt[h : h + 1], obs_gt[0:1]).item())
        pr = float(prediction_mse(prop_gt[h : h + 1], prop_pred[h : h + 1]).item())
        pr_nc = float(prediction_mse(prop_gt[h : h + 1], prop_gt[0:1]).item())
        multi[str(h)] = {
            "visual_mse": vis,
            "visual_mse_no_change": vis_nc,
            "proprio_mse": pr,
            "proprio_mse_no_change": pr_nc,
        }

    return {
        "one_step_visual_mse": one_step_visual,
        "one_step_proprio_mse": one_step_proprio,
        "one_step_visual_mse_no_change": one_step_visual_nc,
        "one_step_proprio_mse_no_change": one_step_proprio_nc,
        "multi_step": multi,
        "finite": bool(
            torch.isfinite(obs_pred).all() and torch.isfinite(prop_pred).all()
        ),
    }


@torch.no_grad()
def representation_diagnostics(
    model: Level1WorldModel,
    images: torch.Tensor,
    proprio: torch.Tensor,
) -> dict[str, Any]:
    """VICReg-style collapse checks on encoded visual maps."""
    model.eval()
    encoded = model.encode_sequence(images, proprio)
    obs = encoded.visual  # [T, B, 16, 43, 43]
    fused = encoded.fused
    flat = flatten_spatial(obs)  # [T, B, D]
    t0 = flat[0]  # [B, D]
    std = t0.std(dim=0, unbiased=False)
    mean_std = float(std.mean().item())
    collapsed = float((std < COLLAPSE_STD_THRESH).float().mean().item())
    # cov_offdiag expects [G, K, D]; use one group = t=0 batch.
    cov = float(cov_offdiag(t0.unsqueeze(0)).item())
    norms = t0.norm(dim=-1)
    return {
        "batch": int(t0.shape[0]),
        "feature_dim": int(t0.shape[1]),
        "visual_shape": list(obs.shape),
        "fused_shape": list(fused.shape),
        "feature_std_mean": mean_std,
        "feature_std_min": float(std.min().item()),
        "feature_std_max": float(std.max().item()),
        "frac_dims_std_lt_1e-3": collapsed,
        "cov_offdiag": cov,
        "feature_norm_mean": float(norms.mean().item()),
        "feature_norm_std": float(norms.std(unbiased=False).item()),
        "finite": bool(torch.isfinite(fused).all() and torch.isfinite(obs).all()),
        "channels": VISUAL_CHANNELS,
        "spatial": FEATURE_SIZE,
    }


@torch.no_grad()
def evaluate_loader_batch(
    model: Level1WorldModel,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    images = batch["images"].to(device).transpose(0, 1)
    proprio = batch["proprio"].to(device).transpose(0, 1)
    actions = batch["actions"].to(device).transpose(0, 1)
    pred = prediction_diagnostics(model, images, proprio, actions)
    rep = representation_diagnostics(model, images, proprio)
    return {"prediction": pred, "representation": rep}
