"""Unit tests for faithful HWM Level-1 objectives (M3)."""

from __future__ import annotations

import pytest
import torch

from hwm_faithful.losses.coefficients import (
    IDM_COEFF,
    PRED_PROPRIO_COEFF,
    VICREG_COV_COEFF,
    VICREG_SIM_COEFF,
    VICREG_STD_COEFF,
    VICREG_STD_COEFF_T,
)
from hwm_faithful.losses.inverse_dynamics import InverseDynamics, idm_loss
from hwm_faithful.losses.level1_objective import Level1Objective
from hwm_faithful.losses.prediction_proprio import prediction_proprio
from hwm_faithful.losses.vicreg_obs import (
    cov_offdiag,
    cov_offdiag_einsum,
    prediction_mse,
    vicreg_obs,
)
from hwm_faithful.models.level1_world_model import Level1WorldModel
from hwm_faithful.models.visual_encoder import count_trainable_parameters
from hwm_faithful.shapes import (
    ACTION_DIM,
    FEATURE_SIZE,
    FUSED_CHANNELS,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    PROPRIO_DIM,
    VISUAL_CHANNELS,
)


def _obs(t: int, b: int, fill: float | None = None) -> torch.Tensor:
    if fill is None:
        return torch.randn(t, b, VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    return torch.full((t, b, VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE), fill)


def _prop(t: int, b: int, fill: float | None = None) -> torch.Tensor:
    if fill is None:
        return torch.randn(t, b, PROPRIO_DIM, FEATURE_SIZE, FEATURE_SIZE)
    return torch.full((t, b, PROPRIO_DIM, FEATURE_SIZE, FEATURE_SIZE), fill)


def _fused(t: int, b: int) -> torch.Tensor:
    return torch.randn(t, b, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)


def test_cov_gram_matches_einsum() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 8, 16)
    a = cov_offdiag(x)
    b = cov_offdiag_einsum(x)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_prediction_mse_perfect_and_perturbed() -> None:
    target = torch.ones(4, 2, 3, 5, 5)
    pred = target.clone()
    assert prediction_mse(target, pred).item() == 0.0
    noisy = pred + 0.5
    assert prediction_mse(target, noisy).item() > 0.0


def test_vicreg_sim_zero_when_pred_matches_gt_from_t1() -> None:
    torch.manual_seed(1)
    obs = _obs(5, 4)
    info = vicreg_obs(obs, obs.clone())
    assert info.sim.item() == 0.0
    assert torch.isfinite(info.total)
    assert info.std.ndim == 0
    assert info.cov.ndim == 0
    # Regularizers are not required to be zero.
    noisy = obs + 1.0
    info_bad = vicreg_obs(obs, noisy)
    assert info_bad.sim.item() > info.sim.item()


def test_vicreg_temporal_indexing_pred_t_vs_gt_t() -> None:
    """Fail if we compared pred[t] to gt[t+1] instead of pred[t+1] to gt[t+1]."""
    t, b = 6, 3
    obs_gt = torch.zeros(t, b, VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    for i in range(t):
        obs_gt[i] = float(i + 1)
    # Correct alignment: pred[k] == gt[k] for all k, including 1:.
    info_ok = vicreg_obs(obs_gt, obs_gt.clone())
    assert info_ok.sim.item() == 0.0
    # Off-by-one: pred[t] = gt[t-1] (and pred[0] dummy).
    obs_shift = torch.zeros_like(obs_gt)
    obs_shift[1:] = obs_gt[:-1]
    info_shift = vicreg_obs(obs_gt, obs_shift)
    assert info_shift.sim.item() > 0.0


def test_prediction_proprio_indexing_and_coeff() -> None:
    t, b = 5, 2
    gt = torch.zeros(t, b, PROPRIO_DIM, FEATURE_SIZE, FEATURE_SIZE)
    for i in range(t):
        gt[i] = float(i)
    info = prediction_proprio(gt, gt.clone())
    assert info.pred_loss.item() == 0.0
    assert info.total.item() == 0.0
    shifted = torch.zeros_like(gt)
    shifted[1:] = gt[:-1]
    info_shift = prediction_proprio(gt, shifted)
    assert info_shift.pred_loss.item() > 0.0
    torch.testing.assert_close(
        info_shift.total, info_shift.pred_loss * PRED_PROPRIO_COEFF
    )


def test_idm_coeff_and_action_alignment() -> None:
    torch.manual_seed(2)
    idm = InverseDynamics()
    t, b = 4, 2
    fused = _fused(t, b)
    actions = torch.randn(t - 1, b, ACTION_DIM)
    info = idm_loss(idm, fused, actions)
    torch.testing.assert_close(info.total, info.action_loss * IDM_COEFF)
    assert torch.isfinite(info.total)
    # Wrong temporal pairing of actions should change the loss.
    shifted = torch.roll(actions, 1, dims=0)
    info_shift = idm_loss(idm, fused, shifted)
    assert not torch.allclose(info.action_loss, info_shift.action_loss)


def test_idm_parameter_count() -> None:
    idm = InverseDynamics()
    n = count_trainable_parameters(idm)
    # Conv36→32 + GN + Conv32→32 + GN + Conv32→32 + GN + Linear(3200,2)
    assert n == 35490
    assert count_trainable_parameters(Level1Objective()) == n


def test_level1_objective_perfect_prediction_components() -> None:
    torch.manual_seed(3)
    obj = Level1Objective()
    t, b = 4, 3
    obs = _obs(t, b)
    prop = _prop(t, b)
    fused = torch.cat([obs, prop], dim=2)
    actions = torch.zeros(t - 1, b, ACTION_DIM)
    out = obj(
        obs_gt=obs,
        obs_pred=obs.clone(),
        proprio_gt=prop,
        proprio_pred=prop.clone(),
        fused_gt=fused,
        actions=actions,
    )
    assert out.vicreg_sim.item() == 0.0
    assert out.prediction_proprio_raw.item() == 0.0
    torch.testing.assert_close(
        out.vicreg_obs,
        VICREG_SIM_COEFF * out.vicreg_sim
        + VICREG_STD_COEFF * out.vicreg_std
        + VICREG_COV_COEFF * out.vicreg_cov
        + VICREG_STD_COEFF_T * out.vicreg_std_t,
    )
    assert torch.isfinite(out.total)
    assert not torch.isnan(out.vicreg_std)
    assert not torch.isinf(out.total)
    torch.testing.assert_close(
        out.total, out.vicreg_obs + out.idm + out.prediction_proprio
    )


def test_level1_objective_repeatable() -> None:
    obj = Level1Objective()
    obj.eval()
    t, b = 3, 2
    torch.manual_seed(4)
    kwargs = dict(
        obs_gt=_obs(t, b),
        obs_pred=_obs(t, b),
        proprio_gt=_prop(t, b),
        proprio_pred=_prop(t, b),
        fused_gt=_fused(t, b),
        actions=torch.randn(t - 1, b, ACTION_DIM),
    )
    with torch.no_grad():
        a = obj(**kwargs)
        b_out = obj(**kwargs)
    torch.testing.assert_close(a.total, b_out.total)


def test_loss_tensors_are_scalars() -> None:
    torch.manual_seed(5)
    obj = Level1Objective()
    t, b = 3, 2
    out = obj(
        obs_gt=_obs(t, b),
        obs_pred=_obs(t, b),
        proprio_gt=_prop(t, b),
        proprio_pred=_prop(t, b),
        fused_gt=_fused(t, b),
        actions=torch.randn(t - 1, b, ACTION_DIM),
    )
    for name in out._fields:
        v = getattr(out, name)
        assert v.ndim == 0, name
        assert v.dtype == torch.float32


def test_objective_grads_into_pred_and_idm() -> None:
    torch.manual_seed(6)
    obj = Level1Objective()
    t, b = 3, 2
    obs_gt = _obs(t, b)
    obs_pred = _obs(t, b).requires_grad_(True)
    prop_gt = _prop(t, b)
    prop_pred = _prop(t, b).requires_grad_(True)
    fused = torch.cat([obs_gt, prop_gt], dim=2).requires_grad_(True)
    actions = torch.randn(t - 1, b, ACTION_DIM)
    out = obj(
        obs_gt=obs_gt,
        obs_pred=obs_pred,
        proprio_gt=prop_gt,
        proprio_pred=prop_pred,
        fused_gt=fused,
        actions=actions,
    )
    out.total.backward()
    assert obs_pred.grad is not None and obs_pred.grad.abs().sum() > 0
    assert prop_pred.grad is not None and prop_pred.grad.abs().sum() > 0
    assert fused.grad is not None and fused.grad.abs().sum() > 0
    idm_grads = [p.grad for p in obj.idm.parameters() if p.requires_grad]
    assert all(g is not None for g in idm_grads)


def test_encoder_predictor_grads_short_window() -> None:
    torch.manual_seed(7)
    wm = Level1WorldModel()
    obj = Level1Objective()
    t, b = 3, 2
    images = torch.randn(t, b, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    proprio = torch.randn(t, b, PROPRIO_DIM)
    actions = torch.randn(t - 1, b, ACTION_DIM)
    encoded, rolled = wm.encode_and_rollout(images, proprio, actions)
    out = obj(
        obs_gt=encoded.visual,
        obs_pred=rolled.obs_component,
        proprio_gt=encoded.proprio_map,
        proprio_pred=rolled.proprio_component,
        fused_gt=encoded.fused,
        actions=actions,
    )
    out.total.backward()
    enc_grads = [p.grad for p in wm.encoder.parameters() if p.requires_grad]
    pred_grads = [p.grad for p in wm.predictor.parameters() if p.requires_grad]
    assert all(g is not None for g in enc_grads)
    assert any(g.abs().sum() > 0 for g in enc_grads)
    assert all(g is not None for g in pred_grads)
    assert any(g.abs().sum() > 0 for g in pred_grads)


def test_invalid_shapes_raise() -> None:
    obj = Level1Objective()
    t, b = 3, 2
    with pytest.raises(ValueError):
        obj(
            obs_gt=_obs(t, b),
            obs_pred=_obs(t, b),
            proprio_gt=_prop(t, b),
            proprio_pred=_prop(t, b),
            fused_gt=_fused(t, b),
            actions=torch.randn(t, b, ACTION_DIM),
        )
    with pytest.raises(ValueError):
        vicreg_obs(_obs(1, 2), _obs(1, 2))
    with pytest.raises(ValueError):
        vicreg_obs(_obs(3, 1), _obs(3, 1))
