"""Tests for faithful HWM Level-2 inputs and latent z (M5)."""

from __future__ import annotations

import pytest
import torch

from hwm_faithful.level2.action_encoder import Level2ActionEncoder
from hwm_faithful.level2.freeze import freeze_module
from hwm_faithful.level2.identity import Level2IdentityEncoder
from hwm_faithful.level2.posterior import Level2ActionPosterior
from hwm_faithful.level2.temporal_abstraction import (
    Level2TemporalConfig,
    build_level2_inputs,
    chunk_level2_actions,
    flatten_action_chunk,
    subsample_level2_states,
)
from hwm_faithful.models.level1_encoder import Level1Encoder
from hwm_faithful.models.visual_encoder import count_trainable_parameters
from hwm_faithful.shapes import (
    ACTION_DIM,
    FEATURE_SIZE,
    FUSED_CHANNELS,
    L2_L1_WINDOW,
    L2_N_STATES,
    L2_N_STEPS,
    L2_PRIMITIVE_HORIZON,
    L2_STEP_SKIP,
    POSTERIOR_INPUT_DIM,
    VISUAL_CHANNELS,
    Z_DIM,
)

# Original PosteriorContinuous + LayerNorm and ConvPredictor 8-64-8 Linears.
EXPECTED_POSTERIOR_PARAMS = 2272
EXPECTED_ACTION_ENCODER_PARAMS = 1096


def _indexed_actions(t: int, batch: int) -> torch.Tensor:
    """action[t, b] = [t + 1000*b, -(t + 1000*b)]."""
    time = torch.arange(t, dtype=torch.float32).view(t, 1, 1)
    batch_off = (1000.0 * torch.arange(batch, dtype=torch.float32)).view(1, batch, 1)
    val = time + batch_off
    return torch.cat([val, -val], dim=-1)


def _indexed_states(t: int, batch: int, *spatial: int) -> torch.Tensor:
    time = torch.arange(t, dtype=torch.float32).view(t, 1, *([1] * len(spatial)))
    return time.expand(t, batch, *spatial).contiguous()


def test_skip10_state_indexing() -> None:
    cfg = Level2TemporalConfig()
    assert cfg.state_indices == (0, 10, 20, 30, 40, 50, 60)
    states = _indexed_states(L2_L1_WINDOW, 2, 3)
    sampled = subsample_level2_states(states, cfg)
    assert sampled.shape == (L2_N_STATES, 2, 3)
    for i, t in enumerate(cfg.state_indices):
        torch.testing.assert_close(sampled[i], torch.full((2, 3), float(t)))


def test_action_chunk_boundaries() -> None:
    cfg = Level2TemporalConfig()
    actions = _indexed_actions(L2_PRIMITIVE_HORIZON, 1)
    chunks = chunk_level2_actions(actions, cfg)
    assert chunks.shape == (L2_N_STEPS, 1, L2_STEP_SKIP, ACTION_DIM)
    for i in range(L2_N_STEPS):
        start, end = i * L2_STEP_SKIP, (i + 1) * L2_STEP_SKIP
        expected = actions[start:end, 0]
        torch.testing.assert_close(chunks[i, 0], expected)
        assert expected[0, 0].item() == float(start)
        assert expected[-1, 0].item() == float(end - 1)


def test_flatten_order_matches_view() -> None:
    actions = _indexed_actions(10, 1)[:, 0]  # [10, 2]
    batched = actions.unsqueeze(0)
    flat = flatten_action_chunk(batched)
    expected = actions.reshape(1, -1)
    torch.testing.assert_close(flat, expected)
    # first pair is t=0, second is t=1
    assert flat[0, 0].item() == 0.0
    assert flat[0, 1].item() == 0.0
    assert flat[0, 2].item() == 1.0
    assert flat[0, 3].item() == -1.0
    assert flat.shape == (1, 20)


def test_b1_and_b2_shapes() -> None:
    for b in (1, 3):
        states = _indexed_states(L2_L1_WINDOW, b, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
        actions = _indexed_actions(L2_PRIMITIVE_HORIZON, b)
        out = build_level2_inputs(states, actions)
        assert out.states.shape == (
            L2_N_STATES,
            b,
            FUSED_CHANNELS,
            FEATURE_SIZE,
            FEATURE_SIZE,
        )
        assert out.action_chunks.shape == (L2_N_STEPS, b, L2_STEP_SKIP, ACTION_DIM)
        assert out.flat_action_chunks.shape == (L2_N_STEPS, b, POSTERIOR_INPUT_DIM)


def test_l2_n_steps_6_exact_shapes() -> None:
    out = build_level2_inputs(
        _indexed_states(70, 2, 4),
        _indexed_actions(70, 2),
    )
    assert out.states.shape[0] == 7
    assert out.action_chunks.shape[0] == 6
    assert out.state_indices == (0, 10, 20, 30, 40, 50, 60)
    assert out.chunk_ranges == (
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 40),
        (40, 50),
        (50, 60),
    )


def test_alignment_h_chunk_hnext() -> None:
    states = _indexed_states(L2_L1_WINDOW, 1, 1)
    actions = _indexed_actions(L2_PRIMITIVE_HORIZON, 1)
    out = build_level2_inputs(states, actions)
    for i in range(L2_N_STEPS):
        t0 = i * L2_STEP_SKIP
        t1 = (i + 1) * L2_STEP_SKIP
        assert out.states[i, 0, 0].item() == float(t0)
        assert out.states[i + 1, 0, 0].item() == float(t1)
        assert out.action_chunks[i, 0, 0, 0].item() == float(t0)
        assert out.action_chunks[i, 0, -1, 0].item() == float(t1 - 1)


def test_invalid_sequence_length() -> None:
    with pytest.raises(ValueError, match="need at least 61"):
        subsample_level2_states(_indexed_states(60, 1, 2))
    with pytest.raises(ValueError, match="need at least 60"):
        chunk_level2_actions(_indexed_actions(59, 1))


def test_identity_concat_zero_params() -> None:
    enc = Level2IdentityEncoder()
    assert count_trainable_parameters(enc) == 0
    visual = torch.randn(2, VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    proprio = torch.randn(2, 2, FEATURE_SIZE, FEATURE_SIZE)
    fused = enc(visual=visual, proprio_map=proprio)
    assert fused.shape == (2, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    torch.testing.assert_close(fused[:, :VISUAL_CHANNELS], visual)
    torch.testing.assert_close(fused[:, VISUAL_CHANNELS:], proprio)
    torch.testing.assert_close(enc(fused=fused), fused)


def test_posterior_output_shape_and_params() -> None:
    post = Level2ActionPosterior()
    assert count_trainable_parameters(post) == EXPECTED_POSTERIOR_PARAMS
    chunk = torch.randn(4, L2_STEP_SKIP, ACTION_DIM)
    out = post(chunk)
    assert out.z.shape == (4, Z_DIM)
    assert out.mu.shape == (4, Z_DIM)
    assert out.std.shape == (4, Z_DIM)
    torch.testing.assert_close(out.z, out.mu)
    assert (out.std >= 0.05).all()
    flat = flatten_action_chunk(chunk)
    out2 = post(flat)
    torch.testing.assert_close(out.z, out2.z)


def test_deterministic_z() -> None:
    torch.manual_seed(0)
    post = Level2ActionPosterior(stochastic=False)
    chunk = torch.randn(3, 10, 2)
    a = post(chunk)
    b = post(chunk)
    assert torch.equal(a.z, b.z)
    assert torch.equal(a.std, b.std)


def test_action_encoder_shape_and_params() -> None:
    enc = Level2ActionEncoder()
    assert count_trainable_parameters(enc) == EXPECTED_ACTION_ENCODER_PARAMS
    z = torch.randn(5, Z_DIM)
    out = enc(z)
    assert out.vector.shape == (5, Z_DIM)
    assert out.spatial.shape == (5, Z_DIM, FEATURE_SIZE, FEATURE_SIZE)
    # spatial is broadcast of vector
    torch.testing.assert_close(
        out.spatial[:, :, 0, 0],
        out.vector,
    )
    torch.testing.assert_close(out.spatial[:, :, 7, 11], out.vector)


def test_posterior_sequence_and_encoder() -> None:
    post = Level2ActionPosterior()
    enc = Level2ActionEncoder()
    chunks = torch.randn(6, 2, 10, 2)
    z = post.encode_sequence(chunks).z
    assert z.shape == (6, 2, 8)
    encoded = enc.encode_sequence(z)
    assert encoded.vector.shape == (6, 2, 8)
    assert encoded.spatial.shape == (6, 2, 8, 43, 43)


def test_gradient_flow() -> None:
    post = Level2ActionPosterior()
    enc = Level2ActionEncoder()
    chunk = torch.randn(2, 10, 2)
    out = post(chunk)
    (out.z ** 2).sum().backward()
    # z = LayerNorm(mu): mu-path weights get grads. The unused std half of the
    # last linear does not (no KL in released L2 objectives).
    last = post.posterior_net[-1]
    assert last.weight.grad is not None
    assert last.weight.grad[: post.z_dim].abs().sum() > 0
    assert last.weight.grad[post.z_dim :].abs().sum() == 0
    assert post.mu_ln.weight.grad is not None
    post.zero_grad()
    encoded = enc(post(chunk).z)
    encoded.spatial.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in post.parameters()
    )
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for p in enc.mlp.parameters()
    )
    assert count_trainable_parameters(enc.expander) == 0


def test_freeze_l1() -> None:
    encoder = Level1Encoder()
    freeze_module(encoder)
    assert count_trainable_parameters(encoder) == 0
    image = torch.randn(2, 3, 98, 98, requires_grad=True)
    proprio = torch.randn(2, 2, requires_grad=True)
    fused = encoder(image, proprio).fused
    fused.sum().backward()
    assert all(p.grad is None for p in encoder.parameters())
    assert image.grad is not None
    live = Level1Encoder()
    img2 = torch.randn(2, 3, 98, 98)
    pr2 = torch.randn(2, 2)
    live(img2, pr2).fused.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in live.parameters())


def test_nondefault_skip_still_consistent() -> None:
    cfg = Level2TemporalConfig(step_skip=4, n_steps=3)
    states = _indexed_states(cfg.l1_window, 1, 1)
    actions = _indexed_actions(cfg.primitive_horizon, 1)
    out = build_level2_inputs(states, actions, cfg)
    assert out.state_indices == (0, 4, 8, 12)
    assert out.action_chunks.shape == (3, 1, 4, 2)
    assert out.states[2, 0, 0].item() == 8.0
