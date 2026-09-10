#!/usr/bin/env python3
"""CPU smoke for faithful L2 inputs / latent z (M5). No f_H, no training."""

from __future__ import annotations

import torch

from hwm_faithful.level2.action_encoder import Level2ActionEncoder
from hwm_faithful.level2.freeze import freeze_module
from hwm_faithful.level2.identity import Level2IdentityEncoder
from hwm_faithful.level2.posterior import Level2ActionPosterior
from hwm_faithful.level2.temporal_abstraction import build_level2_inputs
from hwm_faithful.models.conv_predictor import split_fused
from hwm_faithful.models.level1_encoder import Level1Encoder
from hwm_faithful.models.visual_encoder import count_trainable_parameters


def main() -> None:
    torch.manual_seed(0)
    t_l1, b = 61, 2
    fused = torch.randn(t_l1, b, 18, 43, 43)
    actions = torch.randn(t_l1 - 1, b, 2)
    # stamp time index into channel 0 so subsample indices are visible
    for t in range(t_l1):
        fused[t, :, 0, 0, 0] = float(t)

    l2 = build_level2_inputs(fused, actions)
    post = Level2ActionPosterior()
    act_enc = Level2ActionEncoder()
    ident = Level2IdentityEncoder()
    visual, proprio = split_fused(l2.states)
    h_l2 = ident(visual=visual, proprio_map=proprio)

    z = post.encode_sequence(l2.action_chunks)
    encoded = act_enc.encode_sequence(z.z)
    z_again = post.encode_sequence(l2.action_chunks)

    print("original L1 fused shape      ", tuple(fused.shape))
    print("sampled L2 state shape       ", tuple(l2.states.shape))
    print("identity H shape             ", tuple(h_l2.shape))
    print("action chunk shape           ", tuple(l2.action_chunks.shape))
    print("flattened action chunk shape ", tuple(l2.flat_action_chunks.shape))
    print("z shape                      ", tuple(z.z.shape))
    print("encoded z vector shape       ", tuple(encoded.vector.shape))
    print("encoded z spatial shape      ", tuple(encoded.spatial.shape))
    print("first sampled L2 indices     ", l2.state_indices)
    print("first action-chunk range     ", l2.chunk_ranges[0])
    print("posterior param count        ", count_trainable_parameters(post))
    print("action encoder param count   ", count_trainable_parameters(act_enc))
    print("identity backbone params     ", count_trainable_parameters(ident))
    print("deterministic z equal        ", bool(torch.equal(z.z, z_again.z)))
    print("z is posterior mean          ", bool(torch.equal(z.z, z.mu)))
    print("std min (z_min_std floor)    ", float(z.std.min()))

    # frozen L1 path
    l1 = Level1Encoder()
    freeze_module(l1)
    images = torch.randn(7 * b, 3, 98, 98)
    proprio_vel = torch.randn(7 * b, 2)
    h = l1(images, proprio_vel).fused
    loss = encoded.spatial.sum() + z.z.sum()
    loss.backward()
    print("L1 trainable after freeze    ", count_trainable_parameters(l1))
    print("L1 grads populated           ", any(p.grad is not None for p in l1.parameters()))
    print("posterior has grads          ", all(p.grad is not None for p in post.parameters()))
    print("action encoder mlp has grads ", all(p.grad is not None for p in act_enc.mlp.parameters()))
    print("encoded spatial finite       ", bool(torch.isfinite(encoded.spatial).all()))
    print("h from frozen L1 shape       ", tuple(h.shape))


if __name__ == "__main__":
    main()
