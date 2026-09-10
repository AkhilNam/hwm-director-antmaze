#!/usr/bin/env python3
"""CPU smoke for faithful Level-1 MPPI (M4). Random weights; small K, T."""

from __future__ import annotations

import torch

from hwm_faithful.models.level1_world_model import Level1WorldModel
from hwm_faithful.planning.config import Level1MPPIConfig
from hwm_faithful.planning.level1_planner import Level1MPPIPlanner
from hwm_faithful.planning.mppi import MPPI, bound_plan


def _toy_cost(start: torch.Tensor, target: torch.Tensor):
    def fn(actions: torch.Tensor) -> torch.Tensor:
        k = actions.shape[0]
        state = start.to(actions).unsqueeze(0).expand(k, -1)
        costs = []
        for t in range(actions.shape[1]):
            state = state + actions[:, t]
            costs.append((state - target.to(actions)).pow(2).mean(dim=-1))
        return torch.stack(costs, dim=0).sum(dim=0)

    return fn


def main() -> None:
    print("=== toy additive dynamics (independent of HWM) ===")
    torch.manual_seed(0)
    cfg = Level1MPPIConfig(
        noise_sigma=1.0, num_samples=48, lambda_=0.05, min_step=0.0, max_step=1.0
    )
    ctrl = MPPI(cfg, horizon=3)
    ctrl.change_horizon(3)
    ctrl.U.zero_()
    start = torch.zeros(2)
    target = torch.tensor([0.6, 0.0])
    cost_fn = _toy_cost(start, target)
    init = cost_fn(bound_plan(ctrl.U.unsqueeze(0), 0.0, 1.0)).item()
    step = ctrl.command(cost_fn, shift_nominal_trajectory=False)
    final = cost_fn(bound_plan(step.U.unsqueeze(0), 0.0, 1.0)).item()
    print("toy start         ", tuple(start.tolist()))
    print("toy target        ", tuple(target.tolist()))
    print("toy U shape       ", tuple(step.U.shape))
    print("toy initial cost  ", f"{init:.6f}")
    print("toy optimized cost", f"{final:.6f}")
    print("toy bound plan    ", bound_plan(step.U, 0.0, 1.0).tolist())

    print()
    print("=== HWM Level1Predictor random-weight MPPI smoke ===")
    torch.manual_seed(0)
    wm = Level1WorldModel()
    wm.eval()
    hwm_cfg = Level1MPPIConfig(num_samples=8, noise_sigma=5.0, lambda_=0.0025)
    planner = Level1MPPIPlanner(wm.predictor, hwm_cfg)
    h0 = torch.randn(1, 18, 43, 43)
    tgt = torch.randn(1, 16, 43, 43)
    result = planner.plan(h0, tgt, planning_horizon=3)
    diag = result.diagnostics
    cand = diag["per_env"][0]["candidate_actions"]
    print("current fused shape         ", tuple(h0.shape))
    print("target visual shape         ", tuple(tgt.shape))
    print("candidate actions shape     ", tuple(cand.shape), "  # [K, T, 2]")
    print("predicted trajectory shape  ", tuple(result.predicted_trajectory.shape))
    print("predicted visual shape      ", tuple(result.predicted_visual.shape))
    print("initial exec cost           ", float(diag["initial_exec_cost"][0]))
    print("optimized exec cost         ", float(diag["optimized_exec_cost"][0]))
    print("returned action shape       ", tuple(result.action_sequence.shape))
    print("first action                ", result.action_sequence[0, 0].tolist())
    print("n_refinement_iterations     ", diag["n_refinement_iterations"])
    print("shift_nominal_in_plan       ", diag["shift_nominal_in_plan"])
    print("finite actions              ", bool(torch.isfinite(result.action_sequence).all()))
    print("predictor training restored ", wm.predictor.training is False)


if __name__ == "__main__":
    main()
