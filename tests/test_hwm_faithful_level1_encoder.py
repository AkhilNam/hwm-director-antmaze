"""Shape and gradient tests for the faithful HWM Level-1 encoder (M1)."""

from __future__ import annotations

import pytest
import torch

from hwm_faithful.models.level1_encoder import Level1Encoder
from hwm_faithful.models.proprio import ProprioExpander
from hwm_faithful.models.visual_encoder import (
    VisualEncoder,
    count_trainable_parameters,
)
from hwm_faithful.shapes import (
    FEATURE_SIZE,
    FUSED_CHANNELS,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    PROPRIO_DIM,
    VISUAL_CHANNELS,
)


def _make_batch(batch_size: int, *, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.randn(
        batch_size,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    proprio = torch.randn(
        batch_size, PROPRIO_DIM, dtype=torch.float32, requires_grad=requires_grad
    )
    return image, proprio


@pytest.mark.parametrize("batch_size", [1, 8])
def test_level1_encoder_output_shapes(batch_size: int) -> None:
    encoder = Level1Encoder()
    encoder.eval()
    image, proprio = _make_batch(batch_size)
    out = encoder(image, proprio)
    spatial = (batch_size, FEATURE_SIZE, FEATURE_SIZE)
    assert out.visual.shape == (batch_size, VISUAL_CHANNELS, *spatial[1:])
    assert out.proprio_map.shape == (batch_size, PROPRIO_DIM, *spatial[1:])
    assert out.fused.shape == (batch_size, FUSED_CHANNELS, *spatial[1:])
    assert out.visual.dtype == torch.float32
    assert out.proprio_map.dtype == torch.float32
    assert out.fused.dtype == torch.float32


def test_visual_encoder_alone_shape() -> None:
    vis = VisualEncoder()
    image, _ = _make_batch(2)
    visual = vis(image)
    assert visual.shape == (2, VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)


def test_fusion_ordering_visual_then_proprio() -> None:
    encoder = Level1Encoder()
    encoder.eval()
    image, proprio = _make_batch(3)
    with torch.no_grad():
        out = encoder(image, proprio)
    torch.testing.assert_close(out.fused[:, :VISUAL_CHANNELS], out.visual)
    torch.testing.assert_close(out.fused[:, VISUAL_CHANNELS:], out.proprio_map)


def test_proprio_spatially_broadcast() -> None:
    expander = ProprioExpander()
    proprio = torch.tensor([[0.25, -1.5], [3.0, 0.0]], dtype=torch.float32)
    proprio_map = expander(proprio)
    assert proprio_map.shape == (2, PROPRIO_DIM, FEATURE_SIZE, FEATURE_SIZE)
    for b in range(2):
        for c in range(PROPRIO_DIM):
            plane = proprio_map[b, c]
            assert torch.allclose(plane, torch.full_like(plane, proprio[b, c]))


def test_visual_encoder_has_trainable_grads() -> None:
    encoder = Level1Encoder()
    encoder.train()
    image, proprio = _make_batch(4)
    out = encoder(image, proprio)
    loss = out.fused.pow(2).mean()
    loss.backward()
    visual_grads = [
        p.grad for p in encoder.visual_encoder.parameters() if p.requires_grad
    ]
    assert visual_grads, "visual encoder should have trainable parameters"
    assert all(g is not None for g in visual_grads)
    assert any(g.abs().sum() > 0 for g in visual_grads)
    proprio_params = list(encoder.proprio_expander.parameters())
    assert proprio_params == []


def test_invalid_image_shape_raises() -> None:
    encoder = Level1Encoder()
    proprio = torch.zeros(1, PROPRIO_DIM)
    with pytest.raises(ValueError, match="image must have shape"):
        encoder(torch.zeros(1, 3, 64, 64), proprio)
    with pytest.raises(ValueError, match="image must have shape"):
        encoder(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), proprio)


def test_invalid_proprio_shape_raises() -> None:
    encoder = Level1Encoder()
    image = torch.zeros(2, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    with pytest.raises(ValueError, match="proprio must have shape"):
        encoder(image, torch.zeros(2, 4))
    with pytest.raises(ValueError, match="proprio must have shape"):
        encoder(image, torch.zeros(2))


def test_mismatched_batch_size_raises() -> None:
    encoder = Level1Encoder()
    image = torch.zeros(2, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    proprio = torch.zeros(3, PROPRIO_DIM)
    with pytest.raises(ValueError, match="batch sizes must match"):
        encoder(image, proprio)


def test_parameter_counts() -> None:
    encoder = Level1Encoder()
    n_visual = count_trainable_parameters(encoder.visual_encoder)
    n_proprio = count_trainable_parameters(encoder.proprio_expander)
    n_total = count_trainable_parameters(encoder)
    assert n_proprio == 0
    assert n_total == n_visual
    assert n_visual == 33296
