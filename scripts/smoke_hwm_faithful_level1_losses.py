#!/usr/bin/env python3
"""CPU smoke: full L1 encode → rollout → M3 losses → backward. No training."""

from __future__ import annotations

import torch

from hwm_faithful.losses import Level1Objective
from hwm_faithful.models.level1_world_model import Level1WorldModel
from hwm_faithful.models.visual_encoder import count_trainable_parameters


def main() -> None:
    torch.manual_seed(0)
    wm = Level1WorldModel()
    obj = Level1Objective()
    wm.train()
    obj.train()

    t, b = 15, 2
    images = torch.randn(t, b, 3, 98, 98)
    proprio = torch.randn(t, b, 2)
    actions = torch.randn(t - 1, b, 2)
    print(f"encode+rollout T={t} B={b}", flush=True)

    encoded, rolled = wm.encode_and_rollout(images, proprio, actions)
    out = obj(
        obs_gt=encoded.visual,
        obs_pred=rolled.obs_component,
        proprio_gt=encoded.proprio_map,
        proprio_pred=rolled.proprio_component,
        fused_gt=encoded.fused,
        actions=actions,
    )
    print("h_gt           ", tuple(encoded.fused.shape))
    print("h_pred         ", tuple(rolled.fused.shape))
    print("h_pred[0]=h_gt0", bool(torch.equal(rolled.fused[0], encoded.fused[0])))
    print("vicreg_obs     ", float(out.vicreg_obs))
    print("  sim          ", float(out.vicreg_sim))
    print("  std          ", float(out.vicreg_std))
    print("  cov          ", float(out.vicreg_cov))
    print("  std_t        ", float(out.vicreg_std_t))
    print("idm            ", float(out.idm))
    print("  action_mse   ", float(out.idm_action))
    print("pred_proprio   ", float(out.prediction_proprio))
    print("  raw_mse      ", float(out.prediction_proprio_raw))
    print("total          ", float(out.total))
    print(
        "params         "
        f"encoder={count_trainable_parameters(wm.encoder)} "
        f"predictor={count_trainable_parameters(wm.predictor)} "
        f"idm={count_trainable_parameters(obj.idm)}"
    )

    out.total.backward()
    enc_ok = all(p.grad is not None for p in wm.encoder.parameters() if p.requires_grad)
    pred_ok = all(
        p.grad is not None for p in wm.predictor.parameters() if p.requires_grad
    )
    idm_ok = all(p.grad is not None for p in obj.idm.parameters() if p.requires_grad)
    print(f"encoder grads   {'ok' if enc_ok else 'MISSING'}")
    print(f"predictor grads {'ok' if pred_ok else 'MISSING'}")
    print(f"idm grads       {'ok' if idm_ok else 'MISSING'}")


if __name__ == "__main__":
    main()
