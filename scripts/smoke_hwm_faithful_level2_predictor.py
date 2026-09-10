#!/usr/bin/env python3
"""CPU smoke: M5 skip-10 path + M6 f_H rollout and L2 losses. No training."""

from __future__ import annotations

import torch

from hwm_faithful.level2.freeze import freeze_module
from hwm_faithful.level2.temporal_abstraction import build_level2_inputs
from hwm_faithful.level2.world_model import Level2WorldModel
from hwm_faithful.losses.level2_objective import level2_prediction_losses
from hwm_faithful.models.level1_encoder import Level1Encoder
from hwm_faithful.models.visual_encoder import count_trainable_parameters
from hwm_faithful.shapes import (
    FEATURE_SIZE,
    FUSED_CHANNELS,
    L2_L1_WINDOW,
    L2_PRIMITIVE_HORIZON,
)


def main() -> None:
    torch.manual_seed(0)
    t_l1, b = L2_L1_WINDOW, 2
    fused = torch.randn(t_l1, b, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    actions = torch.randn(L2_PRIMITIVE_HORIZON, b, 2)
    for t in range(t_l1):
        fused[t, :, 0, 0, 0] = float(t)

    l2 = build_level2_inputs(fused, actions)
    l1 = Level1Encoder()
    freeze_module(l1)
    wm = Level2WorldModel()
    wm.train()

    fwd = wm.forward_train(l2.states, l2.action_chunks)
    losses = level2_prediction_losses(fwd.h_gt, fwd.h_pred)
    losses.total.backward()

    conv_params = count_trainable_parameters(wm.predictor.blocks)
    action_params = count_trainable_parameters(wm.predictor.action_encoder)
    f_h_params = count_trainable_parameters(wm.predictor)
    post_params = count_trainable_parameters(wm.posterior)

    last = wm.posterior.posterior_net[-1]
    mu_grad = float(last.weight.grad[: wm.posterior.z_dim].abs().sum())
    std_grad = float(last.weight.grad[wm.posterior.z_dim :].abs().sum())

    print("L1 fused shape               ", tuple(fused.shape))
    print("L2 H_gt shape                ", tuple(fwd.h_gt.shape))
    print("action chunk shape           ", tuple(l2.action_chunks.shape))
    print("z shape                      ", tuple(fwd.z.shape))
    print("posterior mu shape           ", tuple(fwd.posterior_mu.shape))
    print("posterior std shape          ", tuple(fwd.posterior_std.shape))
    print("H_pred shape                 ", tuple(fwd.h_pred.shape))
    print("obs component shape          ", tuple(fwd.rollout.obs_component.shape))
    print("proprio component shape      ", tuple(fwd.rollout.proprio_component.shape))
    print("H_pred[0] equals H_gt[0]     ", bool(torch.equal(fwd.h_pred[0], fwd.h_gt[0])))
    print("z equals posterior mu        ", bool(torch.equal(fwd.z, fwd.posterior_mu)))
    print("action encoder params        ", action_params)
    print("conv predictor params        ", conv_params)
    print("f_H params                   ", f_h_params)
    print("posterior params             ", post_params)
    print("prediction_obs               ", float(losses.prediction_obs))
    print("prediction_proprio           ", float(losses.prediction_proprio))
    print("total L2 loss                ", float(losses.total))
    print("L1 trainable after freeze    ", count_trainable_parameters(l1))
    print("L1 grads populated           ", any(p.grad is not None for p in l1.parameters()))
    print("predictor has grads          ", all(p.grad is not None for p in wm.predictor.blocks.parameters()))
    print("action encoder mlp has grads ", all(p.grad is not None for p in wm.predictor.action_encoder.mlp.parameters()))
    print("posterior mu-half |grad|     ", mu_grad)
    print("posterior std-half |grad|    ", std_grad)
    print("losses finite                ", bool(torch.isfinite(losses.total)))


if __name__ == "__main__":
    main()
