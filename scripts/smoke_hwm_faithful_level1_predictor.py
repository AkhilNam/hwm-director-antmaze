#!/usr/bin/env python3
"""CPU smoke test for the faithful HWM Level-1 predictor (M2). No training."""

from __future__ import annotations

import torch

from hwm_faithful.models.conv_predictor import layer_sequence_description
from hwm_faithful.models.level1_world_model import Level1WorldModel
from hwm_faithful.models.visual_encoder import count_trainable_parameters


def main() -> None:
    torch.manual_seed(0)
    wm = Level1WorldModel()
    wm.train()

    h0 = torch.randn(4, 18, 43, 43, dtype=torch.float32)
    actions = torch.randn(14, 4, 2, dtype=torch.float32)

    one = wm.predict(h0, actions[0])
    rolled = wm.rollout(h0, actions)

    print("h0            ", tuple(h0.shape), h0.dtype)
    print("actions       ", tuple(actions.shape), actions.dtype)
    print("one-step fused", tuple(one.fused.shape))
    print("one-step obs  ", tuple(one.obs_component.shape))
    print("one-step prop ", tuple(one.proprio_component.shape))
    print("rollout fused ", tuple(rolled.fused.shape))
    print("rollout[0]=h0 ", bool(torch.equal(rolled.fused[0], h0)))
    print()
    print("layer sequence:")
    for line in layer_sequence_description():
        print(" ", line)
    print()
    print("intermediate shapes (batch=1 zeros):")
    for name, shape in wm.predictor.intermediate_shapes():
        print(f"  {name}: {shape}")
    print()
    n_pred = count_trainable_parameters(wm.predictor)
    n_act = count_trainable_parameters(wm.predictor.action_encoder)
    n_enc = count_trainable_parameters(wm.encoder)
    n_all = count_trainable_parameters(wm)
    print(
        f"trainable params  predictor={n_pred}  action={n_act}  "
        f"encoder={n_enc}  world_model={n_all}"
    )

    loss = rolled.fused[-1].pow(2).mean()
    loss.backward()
    grads_ok = all(
        p.grad is not None for p in wm.predictor.parameters() if p.requires_grad
    )
    print(f"scalar loss       {loss.item():.6f}")
    print(f"predictor grads   {'ok' if grads_ok else 'MISSING'}")

    image = torch.randn(4, 3, 98, 98, dtype=torch.float32)
    proprio = torch.randn(4, 2, dtype=torch.float32)
    encoded, predicted = wm.encode_and_predict(image, proprio, actions[0])
    print("encode fused     ", tuple(encoded.fused.shape))
    print("pred from encode ", tuple(predicted.fused.shape))


if __name__ == "__main__":
    main()
