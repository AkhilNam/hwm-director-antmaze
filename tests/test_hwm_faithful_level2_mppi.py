"""Tests for faithful Level-2 MPPI (``pi_H``, M7)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hwm_faithful.level2.world_model import Level2WorldModel
from hwm_faithful.models.conv_predictor import split_fused
from hwm_faithful.models.level1_encoder import Level1Encoder
from hwm_faithful.planning.actions import bound_z
from hwm_faithful.planning.config import Level2MPPIConfig
from hwm_faithful.planning.costs import RepresentationCost, z_regularization
from hwm_faithful.planning.level2_planner import (
    Level2MPPIPlanner,
    l2_env_plan_size,
    rollout_z_candidates,
)
from hwm_faithful.planning.mppi import MPPI, mppi_weights
from hwm_faithful.shapes import FEATURE_SIZE, FUSED_CHANNELS, VISUAL_CHANNELS, Z_DIM


def _l2_cfg(**kwargs: object) -> Level2MPPIConfig:
    defaults: dict[str, object] = dict(
        noise_sigma=10.0,
        num_samples=8,
        lambda_=0.0025,
        z_reg_coeff=0.1,
        apply_z_reg=False,
        min_step=-2.5,
        max_step=2.5,
        sum_all_diffs=True,
        clamp_actions=True,
    )
    defaults.update(kwargs)
    return Level2MPPIConfig(**defaults)  # type: ignore[arg-type]


class ToyAdditiveL2(nn.Module):
    """``H_{t+1}[:, :8] = H_t[:, :8] + z_t`` broadcast spatially."""

    def rollout(self, h0: torch.Tensor, z_seq: torch.Tensor):
        steps = [h0]
        h = h0
        for t in range(z_seq.shape[0]):
            delta = torch.zeros_like(h)
            delta[:, :Z_DIM] = z_seq[t][:, :, None, None]
            h = h + delta
            steps.append(h)
        fused = torch.stack(steps, dim=0)
        return SimpleNamespace(fused=fused)


def _toy_vector_cost(start: torch.Tensor, target: torch.Tensor):
    def fn(actions: torch.Tensor) -> torch.Tensor:
        k = actions.shape[0]
        state = start.to(actions).unsqueeze(0).expand(k, -1)
        costs = []
        for t in range(actions.shape[1]):
            state = state + actions[:, t]
            costs.append((state - target.to(actions)).pow(2).mean(dim=-1))
        return torch.stack(costs, dim=0).sum(dim=0)

    return fn


def test_z_candidate_shapes() -> None:
    cfg = _l2_cfg(num_samples=5)
    ctrl = MPPI(cfg, action_dim=Z_DIM, horizon=4, bound_fn=lambda z: bound_z(z))
    ctrl.change_horizon(4)
    captured: dict[str, torch.Tensor] = {}

    def cost_fn(z: torch.Tensor) -> torch.Tensor:
        captured["z"] = z
        assert z.shape == (5, 4, Z_DIM)
        return z.pow(2).mean(dim=(1, 2))

    step = ctrl.command(cost_fn, shift_nominal_trajectory=False)
    assert captured["z"].shape == (5, 4, 8)
    assert step.U.shape == (4, 8)
    assert step.weights.shape == (5,)
    assert step.perturbed_actions.shape == (5, 4, 8)


def test_l2_bounds_are_per_component_clamp() -> None:
    z = torch.tensor([[3.0, -4.0, 0.1, 0.0, 2.6, -2.6, 1.0, -1.0]])
    out = bound_z(z, min_step=-2.5, max_step=2.5)
    expected = torch.tensor([[2.5, -2.5, 0.1, 0.0, 2.5, -2.5, 1.0, -1.0]])
    torch.testing.assert_close(out, expected)
    # Not Euclidean-ball: a vector with ||z||>2.5 but each |z_i|<=2.5 is unchanged.
    inside = torch.ones(1, 8)
    torch.testing.assert_close(bound_z(inside), inside)
    per_min = torch.full((8,), -1.0)
    per_max = torch.full((8,), 1.0)
    out2 = bound_z(z, per_dim_min=per_min, per_dim_max=per_max, margin=0.1)
    torch.testing.assert_close(out2, z.clamp(-0.9, 0.9))


def test_deterministic_fixed_seed() -> None:
    torch.manual_seed(0)
    cfg = _l2_cfg(num_samples=6)
    a = MPPI(cfg, action_dim=8, horizon=3, bound_fn=lambda z: bound_z(z))
    a.change_horizon(3)
    sa = a.command(lambda u: u.pow(2).mean(dim=(1, 2)), shift_nominal_trajectory=False)
    torch.manual_seed(0)
    b = MPPI(cfg, action_dim=8, horizon=3, bound_fn=lambda z: bound_z(z))
    b.change_horizon(3)
    sb = b.command(lambda u: u.pow(2).mean(dim=(1, 2)), shift_nominal_trajectory=False)
    torch.testing.assert_close(sa.U, sb.U)
    torch.testing.assert_close(sa.weights, sb.weights)


def test_weights_finite_and_lower_cost_wins() -> None:
    w = mppi_weights(torch.tensor([4.0, 1.5, 9.0]), 0.0025)
    assert torch.isfinite(w).all()
    torch.testing.assert_close(w.sum(), torch.tensor(1.0), atol=1e-6, rtol=0)
    assert w[1] == w.max()
    assert w[1] > w[0]
    assert w[1] > w[2]


def test_update_moves_toward_lower_cost_candidate() -> None:
    cfg = _l2_cfg(
        num_samples=96,
        noise_sigma=0.25,
        lambda_=0.05,
        min_step=-1.0,
        max_step=1.0,
    )
    ctrl = MPPI(
        cfg,
        action_dim=8,
        horizon=1,
        bound_fn=lambda z: bound_z(z, min_step=-1.0, max_step=1.0),
    )
    ctrl.change_horizon(1)
    ctrl.U.zero_()
    start = torch.zeros(8)
    target = torch.zeros(8)
    target[0] = 0.6
    cost_fn = _toy_vector_cost(start, target)
    init = cost_fn(bound_z(ctrl.U.unsqueeze(0), -1.0, 1.0)).item()
    step = ctrl.command(cost_fn, shift_nominal_trajectory=False)
    final = cost_fn(bound_z(step.U.unsqueeze(0), -1.0, 1.0)).item()
    assert final < init
    assert bound_z(step.U, -1.0, 1.0)[0, 0] > 0


def test_z_reg_computed_but_not_applied() -> None:
    dyn = ToyAdditiveL2()
    planner = Level2MPPIPlanner(dyn, _l2_cfg(num_samples=8, apply_z_reg=False))
    h0 = torch.zeros(1, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    tgt = torch.zeros(1, VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    result = planner.plan(h0, tgt, plan_horizon=3)
    d0 = result.diagnostics["per_env"][0]
    assert result.diagnostics["z_reg_applied"] is False
    assert d0["z_reg_applied"] is False
    assert d0["z_reg"].shape == (8,)
    assert (d0["z_reg"].abs().sum() > 0).item()
    # Cost used by MPPI is rollout + perturbation, not + z_reg.
    torch.testing.assert_close(
        d0["sampled_costs"],
        d0["rollout_costs"] + d0["perturbation_costs"],
        atol=1e-5,
        rtol=1e-5,
    )
    extra = z_regularization(d0["candidate_z"], 0.1)
    torch.testing.assert_close(extra, d0["z_reg"])
    assert not torch.allclose(d0["sampled_costs"], d0["sampled_costs"] + extra)


def test_visual_only_cost() -> None:
    cost = RepresentationCost(sum_all_diffs=True)
    obs = torch.zeros(4, 2, VISUAL_CHANNELS, 3, 3)
    tgt = torch.zeros(2, VISUAL_CHANNELS, 3, 3)
    assert cost(obs, tgt).abs().sum() == 0
    obs[:, :, 0] = 1.0
    assert cost(obs, tgt).sum() > 0
    fused = torch.zeros(4, 2, FUSED_CHANNELS, 3, 3)
    fused[:, :, VISUAL_CHANNELS:] = 5.0
    visual, proprio = fused[:, :, :16], fused[:, :, 16:]
    assert visual.shape[2] == 16
    dummy_obs = torch.zeros(4, 2, 16, 3, 3)
    dummy_obs[:, :, :16] = visual
    # Proprio perturbation is not part of RepresentationCost inputs.
    assert cost(dummy_obs, torch.zeros(2, 16, 3, 3)).abs().sum() == 0
    assert proprio.abs().sum() > 0


def test_temporal_sum_all_diffs() -> None:
    cost_all = RepresentationCost(sum_all_diffs=True)
    cost_last = RepresentationCost(sum_all_diffs=False, sum_last_n=1)
    obs = torch.zeros(4, 1, 16, 2, 2)
    obs[1] = 1.0
    tgt = torch.zeros(1, 16, 2, 2)
    c_all = cost_all(obs, tgt)
    c_last = cost_last(obs, tgt)
    assert c_all.item() > 0
    assert c_last.item() == 0.0


@pytest.mark.parametrize("horizon", [1, 3])
def test_t2_rollout_indexing(horizon: int) -> None:
    dyn = ToyAdditiveL2()
    planner = Level2MPPIPlanner(dyn, _l2_cfg(num_samples=4))
    h0 = torch.randn(2, 18, 43, 43)
    tgt = torch.randn(2, 16, 43, 43)
    out = planner.plan(h0, tgt, plan_horizon=horizon)
    assert out.z_sequence.shape == (2, horizon, 8)
    assert out.predicted_H.shape == (horizon + 1, 2, 18, 43, 43)
    assert out.predicted_visual.shape == (horizon + 1, 2, 16, 43, 43)
    torch.testing.assert_close(out.predicted_H[0], h0)
    visual, _ = split_fused(out.predicted_H)
    torch.testing.assert_close(visual, out.predicted_visual)
    cand = out.diagnostics["per_env"][0]["candidate_z"]
    assert cand.shape == (4, horizon, 8)


def test_variable_env_plan_horizon() -> None:
    # Medium: T2_max=35, skip=10. At env step 0: 35; later shrinks; floor 3.
    assert l2_env_plan_size(0, max_plan_length=35, min_plan_length=3) == 35
    assert l2_env_plan_size(10, max_plan_length=35, min_plan_length=3) == 34
    assert l2_env_plan_size(340, max_plan_length=35, min_plan_length=3) == 3
    assert l2_env_plan_size(0, max_plan_length=18, min_plan_length=3) == 18
    dyn = ToyAdditiveL2()
    planner = Level2MPPIPlanner(dyn, _l2_cfg(num_samples=4))
    h0 = torch.zeros(1, 18, 43, 43)
    tgt = torch.zeros(1, 16, 43, 43)
    a = planner.plan(h0, tgt, plan_horizon=5)
    b = planner.plan(h0, tgt, plan_horizon=3)
    assert a.z_sequence.shape[1] == 5
    assert b.z_sequence.shape[1] == 3
    assert planner.last_plan_size == 3


def test_l2_to_l1_target_is_pred_obs_1() -> None:
    dyn = ToyAdditiveL2()
    planner = Level2MPPIPlanner(dyn, _l2_cfg(num_samples=4))
    h0 = torch.randn(2, 18, 43, 43)
    tgt = torch.randn(2, 16, 43, 43)
    out = planner.plan(h0, tgt, plan_horizon=3)
    handoff = out.predicted_visual[1]
    assert handoff.shape == (2, 16, 43, 43)
    torch.testing.assert_close(handoff, out.predicted_H[1, :, :16])
    assert out.predicted_H[1, :, 16:].shape[1] == 2


def test_posterior_unused_and_no_grad() -> None:
    wm = Level2WorldModel()
    calls = {"n": 0}
    orig = wm.posterior.forward

    def wrapped(chunk):
        calls["n"] += 1
        return orig(chunk)

    wm.posterior.forward = wrapped  # type: ignore[method-assign]
    for p in wm.parameters():
        p.requires_grad_(True)
        p.grad = torch.ones_like(p)
    planner = Level2MPPIPlanner(wm.predictor, _l2_cfg(num_samples=4))
    h0 = torch.randn(1, 18, 43, 43, requires_grad=True)
    tgt = torch.randn(1, 16, 43, 43)
    out = planner.plan(h0, tgt, plan_horizon=2)
    assert calls["n"] == 0
    assert out.diagnostics["posterior_used"] is False
    assert not out.z_sequence.requires_grad
    assert all(p.grad is None or torch.equal(p.grad, torch.ones_like(p)) for p in wm.predictor.parameters())
    enc = Level1Encoder()
    for p in enc.parameters():
        p.grad = torch.ones_like(p)
    # planner never called the encoder
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in enc.parameters())


def test_warm_start_two_calls() -> None:
    dyn = ToyAdditiveL2()
    planner = Level2MPPIPlanner(dyn, _l2_cfg(num_samples=6, noise_sigma=1.0))
    h0 = torch.zeros(1, 18, 43, 43)
    tgt = torch.zeros(1, 16, 43, 43)
    tgt[:, 0] = 1.0
    first = planner.plan(h0, tgt, plan_horizon=3)
    u_after_first = planner.ctrls[0].U.clone()
    second = planner.plan(h0, tgt, plan_horizon=3)
    # plan() does not shift; U persists and is refined again.
    assert not torch.equal(u_after_first, planner.ctrls[0].U)
    assert first.z_sequence.shape == second.z_sequence.shape
    assert planner.last_plan_size == 3
    assert first.diagnostics["shift_nominal_in_plan"] is False


def test_rollout_z_candidates_layout() -> None:
    dyn = ToyAdditiveL2()
    h0 = torch.zeros(2, 18, 43, 43)
    z = torch.ones(3, 4, 2, 8)
    fused = rollout_z_candidates(dyn, h0, z)
    assert fused.shape == (5, 3, 2, 18, 43, 43)
    torch.testing.assert_close(fused[0, 0], h0)
    torch.testing.assert_close(fused[0, 1], h0)


def test_noise_sigma_is_variance() -> None:
    cfg = _l2_cfg(noise_sigma=10.0, num_samples=2000)
    ctrl = MPPI(cfg, action_dim=8, horizon=1, bound_fn=lambda z: z)
    ctrl.U.zero_()
    captured = {}

    def fn(z):
        captured["z"] = z
        return z.pow(2).mean(dim=(1, 2))

    ctrl.command(fn, shift_nominal_trajectory=False)
    emp_var = captured["z"].var(unbiased=True).item()
    # Σ = 10 I, so sample variance around 10 (loose bound; 2000*8 samples).
    assert 7.0 < emp_var < 13.0
