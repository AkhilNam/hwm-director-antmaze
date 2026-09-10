#!/usr/bin/env python3
"""CPU smoke: L2 MPPI -> pred_obs[1] -> L1 MPPI. No environment."""

from __future__ import annotations

import torch

from hwm_faithful.level2.world_model import Level2WorldModel
from hwm_faithful.models.level1_world_model import Level1WorldModel
from hwm_faithful.planning.config import Level1MPPIConfig, Level2MPPIConfig
from hwm_faithful.planning.hierarchical_planner import HierarchicalPlanner


def main() -> None:
    torch.manual_seed(0)
    l1_wm = Level1WorldModel()
    l2_wm = Level2WorldModel()
    l1_wm.eval()
    l2_wm.eval()
    l1_cfg = Level1MPPIConfig(num_samples=8, noise_sigma=5.0, lambda_=0.0025)
    l2_cfg = Level2MPPIConfig(num_samples=8, noise_sigma=10.0, lambda_=0.0025)
    planner = HierarchicalPlanner(l1_wm.predictor, l2_wm.predictor, l1_cfg, l2_cfg)

    h0 = torch.randn(1, 18, 43, 43)
    goal = torch.randn(1, 16, 43, 43)
    out = planner.plan(h0, goal, l2_horizon=3, l1_horizon=10)

    print("current H shape             ", tuple(h0.shape))
    print("final goal visual shape     ", tuple(goal.shape))
    print("z plan shape                ", tuple(out.level2.z_sequence.shape))
    print("coarse trajectory shape     ", tuple(out.level2.predicted_H.shape))
    print("chosen L1 target shape      ", tuple(out.l1_target.shape))
    print("primitive plan shape        ", tuple(out.level1.action_sequence.shape))
    print("first primitive action      ", out.level1.action_sequence[0, 0].tolist())
    print("L2 initial cost             ", float(out.level2.diagnostics["initial_exec_cost"][0]))
    print("L2 optimized cost           ", float(out.level2.diagnostics["optimized_exec_cost"][0]))
    print("L1 initial cost             ", float(out.level1.diagnostics["initial_exec_cost"][0]))
    print("L1 optimized cost           ", float(out.level1.diagnostics["optimized_exec_cost"][0]))
    print("handoff                     ", out.diagnostics["handoff"])
    print("L1 horizon                  ", out.diagnostics["l1_horizon"])
    print("L1 target excludes proprio  ", out.diagnostics["l1_target_excludes_proprio"])
    print("posterior_used              ", out.diagnostics["posterior_used"])
    print("finite z                    ", bool(torch.isfinite(out.level2.z_sequence).all()))
    print("finite primitives           ", bool(torch.isfinite(out.level1.action_sequence).all()))

    n_post = {"n": 0}
    orig = l2_wm.posterior.forward

    def wrapped(chunk):
        n_post["n"] += 1
        return orig(chunk)

    l2_wm.posterior.forward = wrapped  # type: ignore[method-assign]
    planner.plan(h0, goal, l2_horizon=3, l1_horizon=10)
    print("posterior calls during plan ", n_post["n"])


if __name__ == "__main__":
    main()
