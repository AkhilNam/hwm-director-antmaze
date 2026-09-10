"""Tests for the faithful hierarchical planner (M7 handoff)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from hwm_faithful.level2.world_model import Level2WorldModel
from hwm_faithful.models.level1_world_model import Level1WorldModel
from hwm_faithful.planning.config import (
    HierarchicalPlannerConfig,
    Level1MPPIConfig,
    Level2MPPIConfig,
)
from hwm_faithful.planning.hierarchical_planner import HierarchicalPlanner
from hwm_faithful.planning.level2_planner import Level2MPPIPlanner
from hwm_faithful.shapes import L2_STEP_SKIP, VISUAL_CHANNELS, Z_DIM


class ToyAdditiveL2(nn.Module):
    def rollout(self, h0: torch.Tensor, z_seq: torch.Tensor):
        steps = [h0]
        h = h0
        for t in range(z_seq.shape[0]):
            delta = torch.zeros_like(h)
            delta[:, :Z_DIM] = z_seq[t][:, :, None, None]
            h = h + delta
            steps.append(h)
        return SimpleNamespace(fused=torch.stack(steps, dim=0))


class ToyAdditiveL1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    def rollout(self, h0: torch.Tensor, actions: torch.Tensor):
        steps = [h0]
        h = h0
        for t in range(actions.shape[0]):
            delta = torch.zeros_like(h)
            delta[:, :2] = actions[t][:, :, None, None]
            h = h + delta
            steps.append(h)
        fused = torch.stack(steps, dim=0)
        return SimpleNamespace(
            fused=fused,
            obs_component=fused[:, :, :16],
            proprio_component=fused[:, :, 16:],
        )


def _small_cfgs() -> tuple[Level1MPPIConfig, Level2MPPIConfig]:
    l1 = Level1MPPIConfig(num_samples=8, noise_sigma=5.0, lambda_=0.0025)
    l2 = Level2MPPIConfig(num_samples=8, noise_sigma=10.0, lambda_=0.0025)
    return l1, l2


def test_handoff_is_pred_obs_1_visual_only() -> None:
    l1, l2 = _small_cfgs()
    planner = HierarchicalPlanner(ToyAdditiveL1(), ToyAdditiveL2(), l1, l2)
    h0 = torch.randn(2, 18, 43, 43)
    goal = torch.randn(2, 16, 43, 43)
    out = planner.plan(h0, goal, l2_horizon=3, l1_horizon=10)
    assert out.l1_target.shape == (2, VISUAL_CHANNELS, 43, 43)
    torch.testing.assert_close(out.l1_target, out.level2.predicted_visual[1])
    torch.testing.assert_close(out.l1_target, out.level2.predicted_H[1, :, :16])
    assert out.diagnostics["l1_target_excludes_proprio"] is True
    assert out.diagnostics["handoff"] == "pred_obs[1]"
    assert out.diagnostics["l1_horizon"] == L2_STEP_SKIP
    assert out.diagnostics["l1_target_is_final_goal"] is False
    assert not torch.allclose(out.l1_target, goal)


def test_l1_horizon_is_step_skip() -> None:
    l1, l2 = _small_cfgs()
    planner = HierarchicalPlanner(
        ToyAdditiveL1(),
        ToyAdditiveL2(),
        l1,
        l2,
        HierarchicalPlannerConfig(l2_step_skip=10),
    )
    h0 = torch.zeros(1, 18, 43, 43)
    goal = torch.zeros(1, 16, 43, 43)
    out = planner.plan(h0, goal, l2_horizon=3)
    assert out.level1.action_sequence.shape == (1, 10, 2)
    assert out.diagnostics["l1_horizon"] == 10


def test_hierarchical_end_to_end_shapes() -> None:
    l1, l2 = _small_cfgs()
    planner = HierarchicalPlanner(ToyAdditiveL1(), ToyAdditiveL2(), l1, l2)
    h0 = torch.randn(1, 18, 43, 43)
    goal = torch.randn(1, 16, 43, 43)
    out = planner.plan(h0, goal, l2_horizon=3, l1_horizon=10)
    assert out.level2.z_sequence.shape == (1, 3, 8)
    assert out.level2.predicted_H.shape == (4, 1, 18, 43, 43)
    assert out.l1_target.shape == (1, 16, 43, 43)
    assert out.level1.action_sequence.shape == (1, 10, 2)
    assert out.level1.predicted_trajectory.shape == (11, 1, 18, 43, 43)


def test_posterior_unused_in_hierarchy() -> None:
    wm = Level2WorldModel()
    calls = {"n": 0}
    orig = wm.posterior.forward

    def wrapped(chunk):
        calls["n"] += 1
        return orig(chunk)

    wm.posterior.forward = wrapped  # type: ignore[method-assign]
    l1, l2 = _small_cfgs()
    l1.num_samples = 4
    l2.num_samples = 4
    planner = HierarchicalPlanner(ToyAdditiveL1(), wm.predictor, l1, l2)
    h0 = torch.randn(1, 18, 43, 43)
    goal = torch.randn(1, 16, 43, 43)
    planner.plan(h0, goal, l2_horizon=2, l1_horizon=2)
    assert calls["n"] == 0
    assert planner.l2_planner.plan(
        h0, goal, plan_horizon=2
    ).diagnostics["posterior_used"] is False


def test_final_transition_uses_final_goal() -> None:
    l1, l2 = _small_cfgs()
    planner = HierarchicalPlanner(ToyAdditiveL1(), ToyAdditiveL2(), l1, l2)
    h0 = torch.zeros(1, 18, 43, 43)
    goal = torch.ones(1, 16, 43, 43)
    final = planner.plan_final_transition(h0, goal, horizon=15)
    assert final.action_sequence.shape == (1, 15, 2)
    torch.testing.assert_close(planner.l1_planner.target_visual, goal)


def test_no_grad_hierarchy() -> None:
    l1_wm = Level1WorldModel()
    l2_wm = Level2WorldModel()
    for p in list(l1_wm.parameters()) + list(l2_wm.parameters()):
        p.requires_grad_(True)
    l1, l2 = _small_cfgs()
    l1.num_samples = 4
    l2.num_samples = 4
    planner = HierarchicalPlanner(l1_wm.predictor, l2_wm.predictor, l1, l2)
    h0 = torch.randn(1, 18, 43, 43, requires_grad=True)
    goal = torch.randn(1, 16, 43, 43)
    out = planner.plan(h0, goal, l2_horizon=2, l1_horizon=2)
    assert not out.level2.z_sequence.requires_grad
    assert not out.level1.action_sequence.requires_grad
    assert all(p.grad is None for p in l1_wm.parameters())
    assert all(p.grad is None for p in l2_wm.parameters())
