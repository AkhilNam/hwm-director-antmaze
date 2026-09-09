"""Shape, residual, and rollout tests for the faithful HWM Level-1 predictor (M2)."""

from __future__ import annotations

import pytest
import torch

from hwm_faithful.models.action_encoder import PrimitiveActionEncoder
from hwm_faithful.models.conv_predictor import Level1Predictor, split_fused
from hwm_faithful.models.level1_world_model import Level1WorldModel
from hwm_faithful.models.visual_encoder import count_trainable_parameters
from hwm_faithful.shapes import (
    ACTION_DIM,
    FEATURE_SIZE,
    FUSED_CHANNELS,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    PREDICTOR_IN_CHANNELS,
    PROPRIO_DIM,
    VISUAL_CHANNELS,
)


def _make_h(batch_size: int) -> torch.Tensor:
    return torch.randn(
        batch_size, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE, dtype=torch.float32
    )


def _make_action(batch_size: int) -> torch.Tensor:
    return torch.randn(batch_size, ACTION_DIM, dtype=torch.float32)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_one_step_shapes(batch_size: int) -> None:
    predictor = Level1Predictor()
    predictor.eval()
    h = _make_h(batch_size)
    action = _make_action(batch_size)
    out = predictor(h, action)
    spatial = (FEATURE_SIZE, FEATURE_SIZE)
    assert out.fused.shape == (batch_size, FUSED_CHANNELS, *spatial)
    assert out.obs_component.shape == (batch_size, VISUAL_CHANNELS, *spatial)
    assert out.proprio_component.shape == (batch_size, PROPRIO_DIM, *spatial)
    assert out.delta.shape == out.fused.shape
    assert out.fused.dtype == torch.float32
    assert out.obs_component.dtype == torch.float32


def test_component_slicing_visual_then_proprio() -> None:
    predictor = Level1Predictor()
    predictor.eval()
    h = _make_h(3)
    action = _make_action(3)
    with torch.no_grad():
        out = predictor(h, action)
    torch.testing.assert_close(out.fused[:, :VISUAL_CHANNELS], out.obs_component)
    torch.testing.assert_close(out.fused[:, VISUAL_CHANNELS:], out.proprio_component)
    obs, proprio = split_fused(out.fused)
    torch.testing.assert_close(obs, out.obs_component)
    torch.testing.assert_close(proprio, out.proprio_component)


def test_residual_is_h_plus_delta_on_all_channels() -> None:
    predictor = Level1Predictor()
    predictor.eval()
    h = _make_h(4)
    action = _make_action(4)
    with torch.no_grad():
        out = predictor(h, action)
        delta = predictor.compute_delta(h, action)
    torch.testing.assert_close(out.fused, h + delta)
    torch.testing.assert_close(out.delta, delta)
    # Residual is not visual-only: proprio channels also receive delta.
    proprio_delta = delta[:, VISUAL_CHANNELS:]
    assert proprio_delta.abs().sum() > 0


def test_action_dependence() -> None:
    predictor = Level1Predictor()
    predictor.eval()
    h = _make_h(2)
    a1 = torch.zeros(2, ACTION_DIM)
    a2 = torch.ones(2, ACTION_DIM)
    with torch.no_grad():
        y1 = predictor(h, a1).fused
        y2 = predictor(h, a2).fused
    assert not torch.allclose(y1, y2)


def test_action_map_broadcast() -> None:
    encoder = PrimitiveActionEncoder()
    action = torch.tensor([[0.5, -2.0], [1.0, 0.0]], dtype=torch.float32)
    action_map = encoder(action)
    assert action_map.shape == (2, ACTION_DIM, FEATURE_SIZE, FEATURE_SIZE)
    for b in range(2):
        for c in range(ACTION_DIM):
            plane = action_map[b, c]
            assert torch.allclose(plane, torch.full_like(plane, action[b, c]))


def test_predictor_grads() -> None:
    predictor = Level1Predictor()
    predictor.train()
    h = _make_h(4)
    action = _make_action(4)
    out = predictor(h, action)
    loss = out.fused.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in predictor.parameters() if p.requires_grad]
    assert grads
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    assert count_trainable_parameters(predictor.action_encoder) == 0


@pytest.mark.parametrize("horizon", [1, 5, 14])
def test_rollout_shapes_and_recursion(horizon: int) -> None:
    predictor = Level1Predictor()
    predictor.eval()
    batch_size = 3
    h0 = _make_h(batch_size)
    actions = torch.randn(horizon, batch_size, ACTION_DIM)
    with torch.no_grad():
        rolled = predictor.rollout(h0, actions)
    assert rolled.fused.shape == (
        horizon + 1,
        batch_size,
        FUSED_CHANNELS,
        FEATURE_SIZE,
        FEATURE_SIZE,
    )
    assert rolled.obs_component.shape[0] == horizon + 1
    assert rolled.obs_component.shape[2] == VISUAL_CHANNELS
    assert rolled.proprio_component.shape[2] == PROPRIO_DIM
    torch.testing.assert_close(rolled.fused[0], h0)

    h = h0
    with torch.no_grad():
        for t in range(horizon):
            h = predictor(h, actions[t]).fused
            torch.testing.assert_close(rolled.fused[t + 1], h)


def test_rollout_is_open_loop_not_teacher_forced() -> None:
    predictor = Level1Predictor()
    predictor.eval()
    h0 = _make_h(2)
    actions = torch.randn(3, 2, ACTION_DIM)
    other_h = _make_h(2)
    with torch.no_grad():
        open_loop = predictor.rollout(h0, actions).fused
        teacher_h1 = predictor(other_h, actions[0]).fused
        from_h0 = predictor(h0, actions[0]).fused
    torch.testing.assert_close(open_loop[1], from_h0)
    assert not torch.allclose(open_loop[1], teacher_h1)


def test_invalid_h_shape_raises() -> None:
    predictor = Level1Predictor()
    action = _make_action(1)
    with pytest.raises(ValueError, match="h must have shape"):
        predictor(torch.zeros(1, 16, FEATURE_SIZE, FEATURE_SIZE), action)
    with pytest.raises(ValueError, match="h must have shape"):
        predictor(torch.zeros(18, FEATURE_SIZE, FEATURE_SIZE), action)


def test_invalid_action_shape_raises() -> None:
    predictor = Level1Predictor()
    h = _make_h(2)
    with pytest.raises(ValueError, match="action must have shape"):
        predictor(h, torch.zeros(2, 4))
    with pytest.raises(ValueError, match="action must have shape"):
        predictor(h, torch.zeros(2))


def test_mismatched_batch_and_bad_rollout_shapes_raise() -> None:
    predictor = Level1Predictor()
    h = _make_h(2)
    with pytest.raises(ValueError, match="batch sizes must match"):
        predictor(h, _make_action(3))
    with pytest.raises(ValueError, match="actions must have shape"):
        predictor.rollout(h, torch.zeros(2, ACTION_DIM))
    with pytest.raises(ValueError, match="batch sizes must match"):
        predictor.rollout(h, torch.zeros(4, 3, ACTION_DIM))


def test_parameter_count() -> None:
    predictor = Level1Predictor()
    n_action = count_trainable_parameters(predictor.action_encoder)
    n_total = count_trainable_parameters(predictor)
    assert n_action == 0
    # Conv 20→32 3x3 + bias, GN(32)*2, Conv 32→32 3x3 + bias, GN(32)*2,
    # Conv 32→18 3x3 + bias. See docs Section 18.
    assert n_total == 20370


def test_encoder_then_predictor_smoke_path() -> None:
    wm = Level1WorldModel()
    wm.eval()
    image = torch.randn(2, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    proprio = torch.randn(2, PROPRIO_DIM)
    action = _make_action(2)
    with torch.no_grad():
        encoded, predicted = wm.encode_and_predict(image, proprio, action)
    assert encoded.fused.shape == (2, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    assert predicted.fused.shape == encoded.fused.shape
    assert PREDICTOR_IN_CHANNELS == 20
