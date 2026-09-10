#!/usr/bin/env python3
"""CPU smoke for faithful Level-2 MPPI (M7). Toy dynamics + random f_H."""

from __future__ import annotations

import torch

from hwm_faithful.level2.world_model import Level2WorldModel
from hwm_faithful.planning.actions import bound_z
from hwm_faithful.planning.config import Level2MPPIConfig
from hwm_faithful.planning.level2_planner import Level2MPPIPlanner
from hwm_faithful.planning.mppi import MPPI


def _toy_cost(start: torch.Tensor, target: torch.Tensor):
    def fn(z: torch.Tensor) -> torch.Tensor:
        k = z.shape[0]
        state = start.to(z).unsqueeze(0).expand(k, -1)
        costs = []
        for t in range(z.shape[1]):
            state = state + z[:, t]
            costs.append((state - target.to(z)).pow(2).mean(dim=-1))
        return torch.stack(costs, dim=0).sum(dim=0)

    return fn


def main() -> None:
    print("=== toy additive z dynamics (independent of f_H) ===")
    torch.manual_seed(0)
    cfg = Level2MPPIConfig(
        noise_sigma=0.25, num_samples=64, lambda_=0.05, min_step=-2.5, max_step=2.5
    )
    ctrl = MPPI(cfg, action_dim=8, horizon=1, bound_fn=lambda z: bound_z(z))
    ctrl.change_horizon(1)
    ctrl.U.zero_()
    start = torch.zeros(8)
    target = torch.zeros(8)
    target[0] = 1.2
    cost_fn = _toy_cost(start, target)
    init = float(cost_fn(bound_z(ctrl.U.unsqueeze(0))))
    step = ctrl.command(cost_fn, shift_nominal_trajectory=False)
    final = float(cost_fn(bound_z(step.U.unsqueeze(0))))
    print("toy U shape        ", tuple(step.U.shape))
    print("toy initial cost   ", f"{init:.6f}")
    print("toy optimized cost ", f"{final:.6f}")
    print("toy bound z[0]     ", bound_z(step.U)[0].tolist())

    print()
    print("=== random f_H L2 MPPI smoke ===")
    torch.manual_seed(0)
    wm = Level2WorldModel()
    wm.eval()
    hwm_cfg = Level2MPPIConfig(num_samples=8, noise_sigma=10.0, lambda_=0.0025)
    planner = Level2MPPIPlanner(wm.predictor, hwm_cfg)
    h0 = torch.randn(1, 18, 43, 43)
    tgt = torch.randn(1, 16, 43, 43)
    result = planner.plan(h0, tgt, plan_horizon=3)
    diag = result.diagnostics
    cand = diag["per_env"][0]["candidate_z"]
    print("current H shape             ", tuple(h0.shape))
    print("goal visual shape           ", tuple(tgt.shape))
    print("candidate z shape           ", tuple(cand.shape), "  # [K, T2, 8]")
    print("z plan shape                ", tuple(result.z_sequence.shape))
    print("coarse trajectory shape     ", tuple(result.predicted_H.shape))
    print("predicted visual shape      ", tuple(result.predicted_visual.shape))
    print("initial exec cost           ", float(diag["initial_exec_cost"][0]))
    print("optimized exec cost         ", float(diag["optimized_exec_cost"][0]))
    print("first z                     ", result.z_sequence[0, 0].tolist())
    print("n_refinement_iterations     ", diag["n_refinement_iterations"])
    print("posterior_used              ", diag["posterior_used"])
    print("z_reg_applied               ", diag["z_reg_applied"])
    print("finite z                    ", bool(torch.isfinite(result.z_sequence).all()))
    print("finite trajectory           ", bool(torch.isfinite(result.predicted_H).all()))
    print("predictor training restored ", wm.predictor.training is False)
    print("L2 z in [-2.5, 2.5]         ", bool(
        (result.z_sequence >= -2.5).all() and (result.z_sequence <= 2.5).all()
    ))


if __name__ == "__main__":
    main()
