"""Shape, residual, rollout, and loss tests for faithful L2 predictor ``f_H`` (M6)."""

from __future__ import annotations

import pytest
import torch

from hwm_faithful.level2.freeze import freeze_module
from hwm_faithful.level2.predictor import Level2Predictor
from hwm_faithful.level2.world_model import Level2WorldModel
from hwm_faithful.losses.coefficients import PRED_OBS_COEFF, PRED_PROPRIO_COEFF
from hwm_faithful.losses.level2_objective import level2_prediction_losses
from hwm_faithful.models.conv_predictor import split_fused
from hwm_faithful.models.level1_encoder import Level1Encoder
from hwm_faithful.models.visual_encoder import count_trainable_parameters
from hwm_faithful.shapes import (
    FEATURE_SIZE,
    FUSED_CHANNELS,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    L2_N_STATES,
    L2_N_STEPS,
    L2_PREDICTOR_IN_CHANNELS,
    L2_STEP_SKIP,
    PROPRIO_DIM,
    VISUAL_CHANNELS,
    Z_DIM,
)

EXPECTED_ACTION_ENCODER_PARAMS = 1096
EXPECTED_CONV_PARAMS = 112402
EXPECTED_F_H_PARAMS = EXPECTED_ACTION_ENCODER_PARAMS + EXPECTED_CONV_PARAMS  # 113498
EXPECTED_POSTERIOR_PARAMS = 2272


def _make_h(batch_size: int) -> torch.Tensor:
    return torch.randn(
        batch_size, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE, dtype=torch.float32
    )


def _make_z(batch_size: int) -> torch.Tensor:
    return torch.randn(batch_size, Z_DIM, dtype=torch.float32)


def _conv_param_count(predictor: Level2Predictor) -> int:
    return count_trainable_parameters(predictor.blocks)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_one_step_shapes(batch_size: int) -> None:
    predictor = Level2Predictor()
    predictor.eval()
    h = _make_h(batch_size)
    z = _make_z(batch_size)
    out = predictor(h, z)
    spatial = (FEATURE_SIZE, FEATURE_SIZE)
    assert out.fused.shape == (batch_size, FUSED_CHANNELS, *spatial)
    assert out.obs_component.shape == (batch_size, VISUAL_CHANNELS, *spatial)
    assert out.proprio_component.shape == (batch_size, PROPRIO_DIM, *spatial)
    assert out.delta.shape == out.fused.shape
    assert out.z_map.shape == (batch_size, Z_DIM, *spatial)
    assert out.fused.dtype == torch.float32


def test_obs_proprio_split() -> None:
    predictor = Level2Predictor()
    predictor.eval()
    h = _make_h(3)
    z = _make_z(3)
    with torch.no_grad():
        out = predictor(h, z)
    torch.testing.assert_close(out.fused[:, :VISUAL_CHANNELS], out.obs_component)
    torch.testing.assert_close(out.fused[:, VISUAL_CHANNELS:], out.proprio_component)
    obs, proprio = split_fused(out.fused)
    torch.testing.assert_close(obs, out.obs_component)
    torch.testing.assert_close(proprio, out.proprio_component)
    assert out.obs_component.shape[1] == 16
    assert out.proprio_component.shape[1] == 2


def test_residual_is_h_plus_delta_on_all_channels() -> None:
    predictor = Level2Predictor()
    predictor.eval()
    h = _make_h(4)
    z = _make_z(4)
    with torch.no_grad():
        out = predictor(h, z)
        delta, _ = predictor.compute_delta(h, z)
    torch.testing.assert_close(out.fused, h + delta)
    torch.testing.assert_close(out.delta, delta)
    assert delta[:, VISUAL_CHANNELS:].abs().sum() > 0
    assert delta[:, :VISUAL_CHANNELS].abs().sum() > 0


def test_z_affects_output() -> None:
    predictor = Level2Predictor()
    predictor.eval()
    h = _make_h(2)
    z1 = torch.zeros(2, Z_DIM)
    z2 = torch.ones(2, Z_DIM)
    with torch.no_grad():
        y1 = predictor(h, z1).fused
        y2 = predictor(h, z2).fused
    assert not torch.allclose(y1, y2)


def test_predictor_and_action_encoder_grads() -> None:
    predictor = Level2Predictor()
    predictor.train()
    h = _make_h(4)
    z = _make_z(4)
    out = predictor(h, z)
    out.fused.pow(2).mean().backward()
    assert all(p.grad is not None for p in predictor.blocks.parameters())
    assert any(p.grad.abs().sum() > 0 for p in predictor.blocks.parameters())
    assert all(p.grad is not None for p in predictor.action_encoder.mlp.parameters())
    assert any(p.grad.abs().sum() > 0 for p in predictor.action_encoder.mlp.parameters())


def test_no_grads_into_frozen_l1() -> None:
    l1 = Level1Encoder()
    freeze_module(l1)
    predictor = Level2Predictor()
    images = torch.randn(2, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    proprio = torch.randn(2, 2)
    h = l1(images, proprio).fused
    z = _make_z(2)
    loss = predictor(h, z).fused.pow(2).mean()
    loss.backward()
    assert count_trainable_parameters(l1) == 0
    assert all(p.grad is None for p in l1.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.parameters())


def test_invalid_h_and_z_shapes_raise() -> None:
    predictor = Level2Predictor()
    z = _make_z(1)
    with pytest.raises(ValueError, match="H must be"):
        predictor(torch.zeros(1, 16, FEATURE_SIZE, FEATURE_SIZE), z)
    with pytest.raises(ValueError, match="H must be"):
        predictor(torch.zeros(18, FEATURE_SIZE, FEATURE_SIZE), z)
    h = _make_h(2)
    with pytest.raises(ValueError, match="z must be"):
        predictor(h, torch.zeros(2, 4))
    with pytest.raises(ValueError, match="batch mismatch"):
        predictor(h, _make_z(3))
    with pytest.raises(ValueError, match="z_seq must be"):
        predictor.rollout(h, torch.zeros(2, Z_DIM))
    with pytest.raises(ValueError, match="batch mismatch"):
        predictor.rollout(h, torch.zeros(4, 3, Z_DIM))


def test_parameter_counts() -> None:
    predictor = Level2Predictor()
    n_action = count_trainable_parameters(predictor.action_encoder)
    n_conv = _conv_param_count(predictor)
    n_total = count_trainable_parameters(predictor)
    assert n_action == EXPECTED_ACTION_ENCODER_PARAMS
    assert n_conv == EXPECTED_CONV_PARAMS
    assert n_total == EXPECTED_F_H_PARAMS
    assert L2_PREDICTOR_IN_CHANNELS == 26
    wm = Level2WorldModel()
    assert count_trainable_parameters(wm.posterior) == EXPECTED_POSTERIOR_PARAMS
    assert count_trainable_parameters(wm.predictor) == EXPECTED_F_H_PARAMS


@pytest.mark.parametrize("horizon", [1, 6])
def test_rollout_shapes_and_recursion(horizon: int) -> None:
    predictor = Level2Predictor()
    predictor.eval()
    batch_size = 3
    h0 = _make_h(batch_size)
    z_seq = torch.randn(horizon, batch_size, Z_DIM)
    with torch.no_grad():
        rolled = predictor.rollout(h0, z_seq)
    assert rolled.fused.shape == (
        horizon + 1,
        batch_size,
        FUSED_CHANNELS,
        FEATURE_SIZE,
        FEATURE_SIZE,
    )
    assert rolled.obs_component.shape[2] == VISUAL_CHANNELS
    assert rolled.proprio_component.shape[2] == PROPRIO_DIM
    torch.testing.assert_close(rolled.fused[0], h0)
    h = h0
    with torch.no_grad():
        for t in range(horizon):
            h = predictor(h, z_seq[t]).fused
            torch.testing.assert_close(rolled.fused[t + 1], h)


def test_released_t6_rollout_shape() -> None:
    predictor = Level2Predictor()
    h0 = _make_h(2)
    z_seq = torch.randn(L2_N_STEPS, 2, Z_DIM)
    rolled = predictor.rollout(h0, z_seq)
    assert rolled.fused.shape[0] == L2_N_STATES  # 7


def test_rollout_is_open_loop_not_teacher_forced() -> None:
    predictor = Level2Predictor()
    predictor.eval()
    h0 = _make_h(2)
    other = _make_h(2)
    z_seq = torch.randn(3, 2, Z_DIM)
    with torch.no_grad():
        open_loop = predictor.rollout(h0, z_seq).fused
        from_h0 = predictor(h0, z_seq[0]).fused
        teacher_h1 = predictor(other, z_seq[0]).fused
    torch.testing.assert_close(open_loop[1], from_h0)
    assert not torch.allclose(open_loop[1], teacher_h1)


def test_train_forward_shapes() -> None:
    wm = Level2WorldModel()
    b = 2
    h_gt = torch.randn(L2_N_STATES, b, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    chunks = torch.randn(L2_N_STEPS, b, L2_STEP_SKIP, 2)
    out = wm.forward_train(h_gt, chunks)
    assert out.h_gt.shape == h_gt.shape
    assert out.h_pred.shape == (7, b, 18, 43, 43)
    assert out.z.shape == (6, b, 8)
    assert out.posterior_mu.shape == (6, b, 8)
    assert out.posterior_std.shape == (6, b, 8)
    torch.testing.assert_close(out.h_pred[0], h_gt[0])
    torch.testing.assert_close(out.z, out.posterior_mu)


def test_perfect_prediction_losses_zero() -> None:
    h = torch.randn(7, 2, 18, 43, 43)
    out = level2_prediction_losses(h, h.clone())
    assert out.prediction_obs_raw.item() == 0.0
    assert out.prediction_proprio_raw.item() == 0.0
    assert out.prediction_obs.item() == 0.0
    assert out.prediction_proprio.item() == 0.0
    assert out.total.item() == 0.0


def test_perturbed_obs_only_increases_obs() -> None:
    h_gt = torch.randn(7, 2, 18, 43, 43)
    h_pred = h_gt.clone()
    h_pred[:, :, :VISUAL_CHANNELS] = h_pred[:, :, :VISUAL_CHANNELS] + 0.5
    out = level2_prediction_losses(h_gt, h_pred)
    assert out.prediction_obs_raw.item() > 0.0
    assert out.prediction_proprio_raw.item() == 0.0
    assert out.prediction_obs.item() > 0.0
    assert out.prediction_proprio.item() == 0.0


def test_perturbed_proprio_only_increases_proprio() -> None:
    h_gt = torch.randn(7, 2, 18, 43, 43)
    h_pred = h_gt.clone()
    h_pred[:, :, VISUAL_CHANNELS:] = h_pred[:, :, VISUAL_CHANNELS:] + 0.5
    out = level2_prediction_losses(h_gt, h_pred)
    assert out.prediction_proprio_raw.item() > 0.0
    assert out.prediction_obs_raw.item() == 0.0
    assert out.prediction_proprio.item() > 0.0
    assert out.prediction_obs.item() == 0.0


def test_exact_coefficient_application() -> None:
    h_gt = torch.zeros(7, 1, 18, 43, 43)
    h_pred = torch.zeros_like(h_gt)
    h_pred[1:, :, :VISUAL_CHANNELS] = 2.0
    h_pred[1:, :, VISUAL_CHANNELS:] = 3.0
    out = level2_prediction_losses(h_gt, h_pred)
    torch.testing.assert_close(out.prediction_obs, PRED_OBS_COEFF * out.prediction_obs_raw)
    torch.testing.assert_close(
        out.prediction_proprio, PRED_PROPRIO_COEFF * out.prediction_proprio_raw
    )
    assert PRED_OBS_COEFF == PRED_PROPRIO_COEFF == 2.416154262252218
    torch.testing.assert_close(out.prediction_obs_raw, torch.tensor(4.0))
    torch.testing.assert_close(out.prediction_proprio_raw, torch.tensor(9.0))
    torch.testing.assert_close(out.total, out.prediction_obs + out.prediction_proprio)


def test_pred0_excluded() -> None:
    h_gt = torch.zeros(7, 1, 18, 43, 43)
    h_pred = torch.zeros_like(h_gt)
    h_pred[0] = 100.0
    out = level2_prediction_losses(h_gt, h_pred)
    assert out.total.item() == 0.0
    h_pred[1] = 1.0
    out2 = level2_prediction_losses(h_gt, h_pred)
    assert out2.total.item() > 0.0


def test_time_alignment_uses_matching_indices() -> None:
    h_gt = torch.zeros(4, 1, 18, 43, 43)
    for t in range(4):
        h_gt[t] = float(t)
    h_pred = h_gt.clone()
    assert level2_prediction_losses(h_gt, h_pred).total.item() == 0.0
    shifted = torch.stack([h_gt[0], h_gt[0], h_gt[1], h_gt[2]])
    out = level2_prediction_losses(h_gt, shifted)
    assert out.prediction_obs_raw.item() > 0.0


def test_finite_gradients_through_l2_loss() -> None:
    wm = Level2WorldModel()
    h_gt = torch.randn(7, 2, 18, 43, 43)
    chunks = torch.randn(6, 2, 10, 2)
    fwd = wm.forward_train(h_gt, chunks)
    losses = level2_prediction_losses(fwd.h_gt, fwd.h_pred)
    losses.total.backward()
    assert torch.isfinite(losses.total)
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in wm.predictor.parameters()
    )
    last = wm.posterior.posterior_net[-1]
    assert last.weight.grad is not None
    assert torch.isfinite(last.weight.grad).all()
    # mu half of last linear gets grads; unused std half does not (no KL).
    assert last.weight.grad[: wm.posterior.z_dim].abs().sum() > 0
    assert last.weight.grad[wm.posterior.z_dim :].abs().sum() == 0
    assert wm.posterior.mu_ln.weight.grad is not None
    assert wm.posterior.mu_ln.weight.grad.abs().sum() > 0
