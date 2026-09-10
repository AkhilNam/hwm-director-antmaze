"""VICRegObs as executed by original ``VICRegObjective`` with ``pred_attr='obs'``.

Original: ``pldm/objectives/vicreg.py`` + ``PredictionObjective``
(``pldm/objectives/prediction.py``) used as the similarity term.

Active released terms (``large_diverse_25maps.yaml``):

- ``sim``: MSE(obs_gt[1:], obs_pred[1:]) * ``sim_coeff``  (no detach)
- ``std``: hinge on per-feature std of **encoded** obs at t=0, across batch
- ``cov``: off-diagonal covariance of that same t=0 encoding
- ``std_t``: hinge on per-feature std of encoded obs at t=1..T-1, across time

Inactive (coeff 0): ``sim_t``, ``cov_t``. Projector is Identity.
``std`` / ``cov`` do **not** use predictions; they regularize the encoder.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch.nn import functional as F

from hwm_faithful.losses.coefficients import (
    VICREG_ADJUST_COV,
    VICREG_COV_COEFF,
    VICREG_SIM_COEFF,
    VICREG_STD_COEFF,
    VICREG_STD_COEFF_T,
    VICREG_STD_MARGIN,
    VICREG_STD_MARGIN_T,
    VICREG_VAR_EPS,
)


class VICRegObsInfo(NamedTuple):
    total: torch.Tensor
    sim: torch.Tensor
    std: torch.Tensor
    cov: torch.Tensor
    std_t: torch.Tensor


def flatten_spatial(x: torch.Tensor) -> torch.Tensor:
    """Match original ``flatten_conv_output`` for 5D ``[T, B, C, H, W]``."""
    if x.ndim != 5:
        raise ValueError(f"expected [T, B, C, H, W], got {tuple(x.shape)}")
    t, b = x.shape[:2]
    return x.reshape(t, b, -1)


def prediction_mse(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Original ``PredictionObjective``: ``(encodings - predictions).pow(2).mean()``."""
    if target.shape != pred.shape:
        raise ValueError(
            f"prediction MSE shape mismatch: target {tuple(target.shape)} "
            f"vs pred {tuple(pred.shape)}"
        )
    return (target - pred).pow(2).mean()


def std_hinge(x: torch.Tensor, margin: float) -> torch.Tensor:
    """Original ``std_loss`` without the coeff-zero short-circuit.

    ``x`` is ``[G, K, D]``. Center and take unbiased std over ``K`` (dim=1),
    then ``mean(relu(margin - std), dim=-1)``.
    """
    x = x - x.mean(dim=1, keepdim=True)
    std = torch.sqrt(x.var(dim=1) + VICREG_VAR_EPS)
    return torch.mean(F.relu(margin - std), dim=-1)


def cov_offdiag(x: torch.Tensor, *, adjust: bool = VICREG_ADJUST_COV) -> torch.Tensor:
    """Original ``cov_loss`` scalar-per-group, memory-safe Gram form.

    Original einsum is ``cov = X^T X / (K-1)`` then
    ``(||cov||_F^2 - ||diag||^2) / D [/ (D-1) if adjust_cov]``.
    ``||X^T X||_F^2 = ||X X^T||_F^2``, so this uses the ``K×K`` Gram
    and is algebraically the same (see unit test vs einsum).
    """
    if x.shape[1] < 2:
        raise ValueError(
            f"VICReg covariance needs K>=2 samples along dim=1, got {x.shape[1]}"
        )
    x = x - x.mean(dim=1, keepdim=True)
    k = x.shape[1]
    num_features = x.shape[-1]
    gram = torch.einsum("bki,bli->bkl", x, x)
    fro2 = gram.pow(2).sum(dim=(-1, -2)) / (k - 1) ** 2
    col_sq = x.pow(2).sum(dim=1)
    diag2 = (col_sq / (k - 1)).pow(2).sum(dim=-1)
    cov_loss = (fro2 - diag2) / num_features
    if adjust:
        cov_loss = cov_loss / (num_features - 1)
    return cov_loss


def cov_offdiag_einsum(x: torch.Tensor, *, adjust: bool = VICREG_ADJUST_COV) -> torch.Tensor:
    """Literal original ``einsum('bki,bkj->bij')`` form (materializes D×D)."""
    if x.shape[1] < 2:
        raise ValueError(
            f"VICReg covariance needs K>=2 samples along dim=1, got {x.shape[1]}"
        )
    batch_size = x.shape[1]
    num_features = x.shape[-1]
    x = x - x.mean(dim=1, keepdim=True)
    cov = torch.einsum("bki,bkj->bij", x, x) / (batch_size - 1)
    diagonals = torch.einsum("bii->bi", cov).pow(2).sum(dim=-1)
    cov_loss = (cov.pow(2).sum(dim=[-1, -2]) - diagonals).div(num_features)
    if adjust:
        cov_loss = cov_loss / (num_features - 1)
    return cov_loss


def vicreg_obs(
    obs_gt: torch.Tensor,
    obs_pred: torch.Tensor,
) -> VICRegObsInfo:
    """Compute VICRegObs from encoded and predicted visual maps.

    ``obs_gt`` / ``obs_pred``: ``[T, B, 16, H, W]`` with T>=2, B>=2.
    Indexing matches original: sim uses ``[1:]`` vs ``[1:]``;
    std/cov use flattened ``obs_gt[:1]``; std_t uses ``obs_gt[1:]``.
    """
    if obs_gt.ndim != 5 or obs_pred.ndim != 5:
        raise ValueError(
            f"obs_gt/obs_pred must be [T, B, C, H, W], got "
            f"{tuple(obs_gt.shape)} and {tuple(obs_pred.shape)}"
        )
    if obs_gt.shape != obs_pred.shape:
        raise ValueError(
            f"obs_gt and obs_pred shape mismatch: {tuple(obs_gt.shape)} vs "
            f"{tuple(obs_pred.shape)}"
        )
    t, b = obs_gt.shape[:2]
    if t < 2:
        raise ValueError(f"VICRegObs needs T>=2, got {t}")
    if b < 2:
        raise ValueError(f"VICRegObs needs B>=2 (std/cov over batch), got {b}")

    sim = prediction_mse(obs_gt[1:], obs_pred[1:])
    flat = flatten_spatial(obs_gt)
    std = std_hinge(flat[:1], VICREG_STD_MARGIN).mean()
    cov = cov_offdiag(flat[:1]).mean()
    std_t = std_hinge(flat[1:].permute(1, 0, 2), VICREG_STD_MARGIN_T).mean()

    total = (
        VICREG_SIM_COEFF * sim
        + VICREG_COV_COEFF * cov
        + VICREG_STD_COEFF * std
        + VICREG_STD_COEFF_T * std_t
    )
    return VICRegObsInfo(total=total, sim=sim, std=std, cov=cov, std_t=std_t)
