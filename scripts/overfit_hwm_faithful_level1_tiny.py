#!/usr/bin/env python3
"""Tiny synthetic overfit for M3. Not full training. CPU only."""

from __future__ import annotations

import torch

from hwm_faithful.losses import Level1Objective
from hwm_faithful.models.level1_world_model import Level1WorldModel


def _deterministic_batch(t: int, b: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simple time/batch-indexed tensors the model can memorize."""
    images = torch.zeros(t, b, 3, 98, 98)
    proprio = torch.zeros(t, b, 2)
    actions = torch.zeros(t - 1, b, 2)
    for i in range(t):
        val = (i + 1) / t
        images[i] = val
        proprio[i, :, 0] = val
        proprio[i, :, 1] = -val
        if b > 1:
            images[i, 1] = val + 0.15
            proprio[i, 1, 0] = val + 0.15
    for i in range(t - 1):
        actions[i, :, 0] = 0.25
        actions[i, :, 1] = -0.1
        if b > 1:
            actions[i, 1, 0] = 0.4
    return images, proprio, actions


def _forward(wm, obj, images, proprio, actions):
    encoded, rolled = wm.encode_and_rollout(images, proprio, actions)
    return obj(
        obs_gt=encoded.visual,
        obs_pred=rolled.obs_component,
        proprio_gt=encoded.proprio_map,
        proprio_pred=rolled.proprio_component,
        fused_gt=encoded.fused,
        actions=actions,
    )


def _print(tag: str, out) -> None:
    print(
        f"{tag}  total={out.total.item():.6f}  "
        f"vicreg={out.vicreg_obs.item():.6f}  "
        f"idm={out.idm.item():.6f}  "
        f"prop={out.prediction_proprio.item():.6f}  "
        f"(sim={out.vicreg_sim.item():.6f} std={out.vicreg_std.item():.6f} "
        f"cov={out.vicreg_cov.item():.6f} std_t={out.vicreg_std_t.item():.6f})",
        flush=True,
    )


def main() -> None:
    torch.manual_seed(0)
    t, b = 15, 2
    n_steps = 15
    images, proprio, actions = _deterministic_batch(t, b)
    print(f"tiny overfit T={t} B={b} steps={n_steps} lr=1e-3", flush=True)

    wm = Level1WorldModel()
    obj = Level1Objective()
    wm.train()
    obj.train()
    opt = torch.optim.Adam(
        list(wm.parameters()) + list(obj.parameters()), lr=1e-3
    )

    mid = n_steps // 2
    history = []
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        out = _forward(wm, obj, images, proprio, actions)
        out.total.backward()
        opt.step()
        history.append(
            {
                "step": step,
                "total": float(out.total.detach()),
                "vicreg_obs": float(out.vicreg_obs.detach()),
                "idm": float(out.idm.detach()),
                "prediction_proprio": float(out.prediction_proprio.detach()),
            }
        )
        if step in (0, mid, n_steps - 1):
            _print(f"step {step:02d}", out)

    first, last = history[0]["total"], history[-1]["total"]
    print(f"total {first:.6f} -> {last:.6f}  decreased={last < first}")
    if last >= first:
        raise SystemExit("overfit sanity failed: total loss did not decrease")


if __name__ == "__main__":
    main()
