"""Hierarchical planner: ``pi_H`` then handoff to ``pi_L``.

Original ``TwoLvlPlanner.plan`` (SHA ``e197375``):

1. L2 MPPI over fused ``H`` toward the **final goal visual**
2. L1 target = ``l2_result.pred_obs[1]`` (first predicted L2 visual)
3. L1 MPPI with ``plan_size = l2_step_skip`` (10)

``final_trans_steps=15`` is **not** inside ``TwoLvlPlanner``. The env loop
runs hierarchical MPC for ``n_steps - 15``, then a separate flat L1 MPC
toward the encoded **final goal**. ``plan_final_transition`` is that
planner-side call without an env loop.

``mock_l1`` is false in the released YAML and is not implemented.
``pred_encoder = L2 IdentityEncoder`` on the hierarchical L1 objective is a
no-op on already-visual ``obs_component`` maps; we pass the 16-channel
target directly.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch

from hwm_faithful.models.conv_predictor import Level1Predictor
from hwm_faithful.level2.predictor import Level2Predictor
from hwm_faithful.planning.config import (
    HierarchicalPlannerConfig,
    Level1MPPIConfig,
    Level2MPPIConfig,
)
from hwm_faithful.planning.level1_planner import Level1MPPIPlanner, Level1PlanResult
from hwm_faithful.planning.level2_planner import Level2MPPIPlanner, Level2PlanResult
from hwm_faithful.shapes import L2_STEP_SKIP, VISUAL_CHANNELS


class HierarchicalPlanResult(NamedTuple):
    level2: Level2PlanResult
    level1: Level1PlanResult
    l1_target: torch.Tensor  # [B, 16, 43, 43]
    diagnostics: dict[str, Any]


class HierarchicalPlanner:
    """``pi_H`` then ``pi_L`` with the original ``pred_obs[1]`` handoff."""

    def __init__(
        self,
        l1_predictor: Level1Predictor,
        l2_predictor: Level2Predictor,
        l1_config: Level1MPPIConfig | None = None,
        l2_config: Level2MPPIConfig | None = None,
        h_config: HierarchicalPlannerConfig | None = None,
        *,
        n_envs: int | None = None,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
    ) -> None:
        self.h_config = h_config or HierarchicalPlannerConfig()
        self.l1_planner = Level1MPPIPlanner(
            l1_predictor,
            l1_config,
            n_envs=n_envs,
            action_mean=action_mean,
            action_std=action_std,
        )
        self.l2_planner = Level2MPPIPlanner(
            l2_predictor, l2_config, n_envs=n_envs
        )

    def reset(self) -> None:
        self.l1_planner.reset()
        self.l2_planner.reset()

    def set_l2_target(self, target_visual: torch.Tensor) -> None:
        self.l2_planner.set_target(target_visual)

    @torch.no_grad()
    def plan(
        self,
        current_fused_state: torch.Tensor,
        final_goal_visual: torch.Tensor,
        l2_horizon: int,
        l1_horizon: int | None = None,
    ) -> HierarchicalPlanResult:
        """Hierarchical handoff. ``l1_horizon`` defaults to ``l2_step_skip=10``."""
        if l1_horizon is None:
            l1_horizon = self.h_config.l2_step_skip
        if l1_horizon != self.h_config.l2_step_skip:
            # Allowed for CPU smoke; original TwoLvlPlanner always uses skip.
            pass
        if self.h_config.mock_l1:
            raise NotImplementedError("mock_l1 is unused in released Diverse Maze")

        l2_result = self.l2_planner.plan(
            current_state=current_fused_state,
            target_visual=final_goal_visual,
            plan_horizon=l2_horizon,
        )
        if l2_result.predicted_visual.shape[0] < 2:
            raise ValueError("L2 trajectory must include at least pred_obs[1]")
        l1_target = l2_result.predicted_visual[1].detach()
        if l1_target.shape[1] != VISUAL_CHANNELS:
            raise ValueError(
                f"L1 target must be visual {VISUAL_CHANNELS}ch, got {tuple(l1_target.shape)}"
            )

        l1_result = self.l1_planner.plan(
            current_fused_state=current_fused_state,
            target_visual_representation=l1_target,
            planning_horizon=l1_horizon,
        )
        diagnostics = {
            "handoff": "pred_obs[1]",
            "l1_target_excludes_proprio": True,
            "l1_horizon": l1_horizon,
            "l2_horizon": l2_horizon,
            "l2_step_skip": self.h_config.l2_step_skip,
            "posterior_used": False,
            "l1_target_is_final_goal": False,
        }
        return HierarchicalPlanResult(
            level2=l2_result,
            level1=l1_result,
            l1_target=l1_target,
            diagnostics=diagnostics,
        )

    @torch.no_grad()
    def plan_final_transition(
        self,
        current_fused_state: torch.Tensor,
        final_goal_visual: torch.Tensor,
        horizon: int | None = None,
    ) -> Level1PlanResult:
        """Stage-2 flat L1 MPPI toward the **final goal** encoding.

        Original env loop uses ``final_trans_steps=15`` after hierarchical
        stage 1. No L2 MPPI and no ``pred_obs[1]`` handoff.
        """
        if horizon is None:
            horizon = self.h_config.final_trans_steps
        return self.l1_planner.plan(
            current_fused_state=current_fused_state,
            target_visual_representation=final_goal_visual,
            planning_horizon=horizon,
        )


assert L2_STEP_SKIP == 10
