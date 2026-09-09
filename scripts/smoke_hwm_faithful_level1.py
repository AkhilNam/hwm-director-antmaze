#!/usr/bin/env python3
"""CPU smoke test for the faithful HWM Level-1 encoder (M1). No training."""

from __future__ import annotations

import torch

from hwm_faithful.models.level1_encoder import Level1Encoder
from hwm_faithful.models.visual_encoder import (
    count_trainable_parameters,
    layer_sequence_description,
)


def main() -> None:
    torch.manual_seed(0)
    encoder = Level1Encoder()
    encoder.train()

    image = torch.randn(4, 3, 98, 98, dtype=torch.float32)
    proprio = torch.randn(4, 2, dtype=torch.float32)
    out = encoder(image, proprio)

    print("image        ", tuple(image.shape), image.dtype)
    print("proprio      ", tuple(proprio.shape), proprio.dtype)
    print("visual       ", tuple(out.visual.shape))
    print("proprio_map  ", tuple(out.proprio_map.shape))
    print("fused        ", tuple(out.fused.shape))
    print()
    print("layer sequence:")
    for line in layer_sequence_description():
        print(" ", line)
    print()
    print("intermediate shapes (batch=1 zeros):")
    for name, shape in encoder.visual_encoder.intermediate_shapes():
        print(f"  {name}: {shape}")
    print()
    n_vis = count_trainable_parameters(encoder.visual_encoder)
    n_prop = count_trainable_parameters(encoder.proprio_expander)
    n_all = count_trainable_parameters(encoder)
    print(f"trainable params  visual={n_vis}  proprio={n_prop}  total={n_all}")

    loss = out.fused.pow(2).mean()
    loss.backward()
    grads_ok = all(
        p.grad is not None for p in encoder.visual_encoder.parameters() if p.requires_grad
    )
    print(f"scalar loss       {loss.item():.6f}")
    print(f"visual grads      {'ok' if grads_ok else 'MISSING'}")


if __name__ == "__main__":
    main()
