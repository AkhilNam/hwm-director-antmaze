#!/usr/bin/env python3
"""Tiny synthetic overfit for M6 Level-2 f_H. Not full training. CPU only.

Trains posterior + L2 action encoder + L2 predictor. Does not train L1.
"""

from __future__ import annotations

import torch

from hwm_faithful.level2.world_model import Level2WorldModel
from hwm_faithful.losses.level2_objective import level2_prediction_losses
from hwm_faithful.shapes import FEATURE_SIZE, FUSED_CHANNELS, L2_N_STATES, L2_N_STEPS


def _deterministic_l2_batch(
    t: int, b: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Time/batch-indexed fused states and primitive action chunks."""
    h_gt = torch.zeros(t, b, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    chunks = torch.zeros(t - 1, b, 10, 2)
    for i in range(t):
        val = (i + 1) / t
        h_gt[i, :, :16] = val
        h_gt[i, :, 16] = val
        h_gt[i, :, 17] = -val
        if b > 1:
            h_gt[i, 1, :16] = val + 0.15
            h_gt[i, 1, 16] = val + 0.15
            h_gt[i, 1, 17] = -(val + 0.15)
        yy = torch.linspace(-1.0, 1.0, FEATURE_SIZE).view(1, 1, FEATURE_SIZE, 1)
        h_gt[i, :, 0:1] = h_gt[i, :, 0:1] + 0.05 * yy
    for i in range(t - 1):
        chunks[i, :, :, 0] = 0.25 + 0.05 * i
        chunks[i, :, :, 1] = -0.1
        if b > 1:
            chunks[i, 1, :, 0] = 0.4 + 0.05 * i
    return h_gt, chunks


def _print(tag: str, out) -> None:
    print(
        f"{tag}  total={out.total.item():.6f}  "
        f"obs={out.prediction_obs.item():.6f}  "
        f"proprio={out.prediction_proprio.item():.6f}  "
        f"(obs_raw={out.prediction_obs_raw.item():.6f} "
        f"proprio_raw={out.prediction_proprio_raw.item():.6f})",
        flush=True,
    )


def main() -> None:
    torch.manual_seed(0)
    t, b = L2_N_STATES, 2
    n_steps = 40
    h_gt, chunks = _deterministic_l2_batch(t, b)
    print(f"tiny L2 overfit T_states={t} B={b} steps={n_steps} lr=1e-3", flush=True)
    print("trainable: posterior + L2 action encoder + L2 predictor (no L1)", flush=True)

    wm = Level2WorldModel()
    wm.train()
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)

    mid = n_steps // 2
    history = []
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        fwd = wm.forward_train(h_gt, chunks)
        out = level2_prediction_losses(fwd.h_gt, fwd.h_pred)
        out.total.backward()
        opt.step()
        history.append(
            {
                "step": step,
                "total": float(out.total.detach()),
                "obs": float(out.prediction_obs.detach()),
                "proprio": float(out.prediction_proprio.detach()),
            }
        )
        if step in (0, mid, n_steps - 1):
            _print(f"step {step:02d}", out)

    first, last = history[0]["total"], history[-1]["total"]
    print(f"total {first:.6f} -> {last:.6f}  decreased={last < first}")
    if last >= first:
        raise SystemExit("overfit sanity failed: total L2 loss did not decrease")


if __name__ == "__main__":
    main()
