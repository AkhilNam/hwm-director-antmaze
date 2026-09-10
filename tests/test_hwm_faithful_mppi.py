"""Unit tests for faithful HWM Level-1 MPPI (M4)."""

from __future__ import annotations

import pytest
import torch

from hwm_faithful.models.conv_predictor import Level1Predictor
from hwm_faithful.planning.actions import rescale_action_norm, unnormalize_action
from hwm_faithful.planning.config import Level1MPPIConfig
from hwm_faithful.planning.costs import RepresentationCost, trajectory_repr_cost
from hwm_faithful.planning.level1_planner import (
    Level1MPPIPlanner,
    rollout_action_candidates,
)
from hwm_faithful.planning.mppi import MPPI, bound_plan, mppi_weights, sample_noise
from hwm_faithful.shapes import ACTION_DIM, FEATURE_SIZE, FUSED_CHANNELS, VISUAL_CHANNELS


def _cfg(**kwargs: object) -> Level1MPPIConfig:
    defaults: dict[str, object] = dict(
        noise_sigma=5.0,
        num_samples=8,
        lambda_=0.0025,
        min_step=0.0,
        max_step=1.0,
        sum_all_diffs=True,
        clamp_actions=False,
    )
    defaults.update(kwargs)
    return Level1MPPIConfig(**defaults)  # type: ignore[arg-type]


def _toy_rollout_cost(start: torch.Tensor, target: torch.Tensor):
    """next_state = state + action; cost = sum_t mean((s_t - target)^2)."""

    def fn(actions: torch.Tensor) -> torch.Tensor:
        # actions [K, T, 2]
        k = actions.shape[0]
        state = start.to(actions).unsqueeze(0).expand(k, -1)
        costs = []
        for t in range(actions.shape[1]):
            state = state + actions[:, t]
            costs.append((state - target.to(actions)).pow(2).mean(dim=-1))
        return torch.stack(costs, dim=0).sum(dim=0)

    return fn


def test_candidate_and_action_shapes() -> None:
    cfg = _cfg(num_samples=5)
    ctrl = MPPI(cfg, horizon=4)
    ctrl.change_horizon(4)
    captured: dict[str, torch.Tensor] = {}

    def cost_fn(actions: torch.Tensor) -> torch.Tensor:
        captured["actions"] = actions
        assert actions.shape == (5, 4, ACTION_DIM)
        return actions.pow(2).mean(dim=(1, 2))

    step = ctrl.command(cost_fn, shift_nominal_trajectory=False)
    assert captured["actions"].shape == (5, 4, 2)
    assert step.U.shape == (4, 2)
    assert step.weights.shape == (5,)
    assert step.costs.shape == (5,)
    assert step.noise.shape == (5, 4, 2)
    assert step.perturbed_actions.shape == (5, 4, 2)


def test_action_norm_bounds_not_per_component() -> None:
    long = torch.tensor([[3.0, 4.0]])  # norm 5
    out = rescale_action_norm(long, min_norm=0.0, max_norm=1.0)
    torch.testing.assert_close(out, torch.tensor([[0.6, 0.8]]), atol=1e-5, rtol=0)
    short = torch.tensor([[0.3, 0.4]])  # norm 0.5
    torch.testing.assert_close(
        rescale_action_norm(short, min_norm=0.0, max_norm=1.0),
        short,
        atol=1e-5,
        rtol=0,
    )
    left = torch.tensor([[-2.0, 0.0]])
    out_left = rescale_action_norm(left, min_norm=0.0, max_norm=1.0)
    torch.testing.assert_close(out_left, torch.tensor([[-1.0, 0.0]]), atol=1e-5, rtol=0)
    # min_step=0, max_step=1 is NOT a clip of each component to [0, 1].
    assert out_left[0, 0] < 0
    clamped = rescale_action_norm(
        torch.tensor([[-0.5, 2.0]]), min_norm=0.0, max_norm=1.0, clamp_components=True
    )
    torch.testing.assert_close(clamped, torch.tensor([[0.0, 1.0]]))


def test_identity_unnormalize() -> None:
    a = torch.tensor([[[0.2, -0.3]]])
    torch.testing.assert_close(unnormalize_action(a), a)


def test_deterministic_seed() -> None:
    torch.manual_seed(0)
    cfg = _cfg(num_samples=6)
    a = MPPI(cfg, horizon=3)
    a.change_horizon(3)
    step_a = a.command(lambda u: u.pow(2).mean(dim=(1, 2)), shift_nominal_trajectory=False)
    torch.manual_seed(0)
    b = MPPI(cfg, horizon=3)
    b.change_horizon(3)
    step_b = b.command(lambda u: u.pow(2).mean(dim=(1, 2)), shift_nominal_trajectory=False)
    torch.testing.assert_close(step_a.U, step_b.U)
    torch.testing.assert_close(step_a.weights, step_b.weights)


def test_weights_finite_and_normalized() -> None:
    costs = torch.tensor([4.0, 1.5, 9.0, 1.5])
    w = mppi_weights(costs, 0.0025)
    assert torch.isfinite(w).all()
    torch.testing.assert_close(w.sum(), torch.tensor(1.0), atol=1e-6, rtol=0)
    assert (w >= 0).all()


def test_lower_cost_gets_larger_weight() -> None:
    costs = torch.tensor([10.0, 1.0, 8.0])
    w = mppi_weights(costs, 0.0025)
    assert w[1] > w[0]
    assert w[1] > w[2]
    # lambda is small → almost a hard argmin
    assert w[1].item() > 0.99


def test_manual_k_small_update() -> None:
    """K=2, T=1: inject known costs/noise into the update equation."""
    costs = torch.tensor([2.0, 0.0])
    w = mppi_weights(costs, 1.0)
    # exp(-(2-0)/1)=e^{-2}, exp(0)=1 → w = [e^{-2}, 1] / (1+e^{-2})
    expected_w1 = 1.0 / (1.0 + torch.exp(torch.tensor(-2.0)).item())
    assert abs(w[1].item() - expected_w1) < 1e-6
    noise = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])  # [K, T, 2]
    delta = (w.view(-1, 1, 1) * noise).sum(dim=0)
    u = torch.zeros(1, 2) + delta
    # update should move more in the cheaper sample's noise direction (y)
    assert u[0, 1] > u[0, 0]


def test_plan_moves_toward_low_cost_perturbation() -> None:
    cfg = _cfg(num_samples=32, noise_sigma=1.0, lambda_=0.05)
    ctrl = MPPI(cfg, horizon=1)
    ctrl.change_horizon(1)
    ctrl.U.zero_()
    start = torch.zeros(2)
    target = torch.tensor([0.5, 0.0])
    cost_fn = _toy_rollout_cost(start, target)
    init = cost_fn(bound_plan(ctrl.U.unsqueeze(0), 0.0, 1.0))
    step = ctrl.command(cost_fn, shift_nominal_trajectory=False)
    final = cost_fn(bound_plan(step.U.unsqueeze(0), 0.0, 1.0))
    assert final.item() < init.item()
    # first (only) action should have positive x
    bound_u = bound_plan(step.U, 0.0, 1.0)
    assert bound_u[0, 0] > 0


@pytest.mark.parametrize("horizon", [1, 10])
def test_horizons(horizon: int) -> None:
    cfg = _cfg(num_samples=4)
    ctrl = MPPI(cfg, horizon=horizon)
    ctrl.change_horizon(horizon)
    step = ctrl.command(
        lambda u: u.pow(2).mean(dim=(1, 2)), shift_nominal_trajectory=False
    )
    assert step.U.shape == (horizon, 2)
    assert step.perturbed_actions.shape == (4, horizon, 2)


def test_batch_b1_and_b2_independent_controllers() -> None:
    cfg = _cfg(num_samples=4)
    start = torch.zeros(2)
    target_a = torch.tensor([0.4, 0.0])
    target_b = torch.tensor([0.0, 0.4])
    ctrls = [MPPI(cfg, horizon=1) for _ in range(2)]
    for c in ctrls:
        c.change_horizon(1)
        c.U.zero_()
    torch.manual_seed(1)
    ua = ctrls[0].command(_toy_rollout_cost(start, target_a)).U
    ub = ctrls[1].command(_toy_rollout_cost(start, target_b)).U
    assert ua.shape == (1, 2)
    assert ub.shape == (1, 2)
    # different targets → different plans (very likely with peaked lambda)
    assert not torch.allclose(ua, ub, atol=1e-3)


def test_sample_noise_layout() -> None:
    noise = sample_noise(
        7, 3, 2, 5.0, device=torch.device("cpu"), dtype=torch.float32
    )
    assert noise.shape == (7, 3, 2)


def test_repr_cost_exact_match() -> None:
    target = torch.ones(2, 2, 2)
    traj = torch.ones(5, 3, 2, 2, 2)
    cost = trajectory_repr_cost(traj, target, sum_all_diffs=True)
    torch.testing.assert_close(cost, torch.zeros(3))


def test_repr_cost_constant_offset() -> None:
    target = torch.zeros(1, 2, 2, 2)
    traj = torch.full((4, 2, 2, 2, 2), 0.5)  # T_pred=3
    cost = trajectory_repr_cost(traj, target, sum_all_diffs=True)
    # mean((0.5)^2)=0.25 per step, 3 predicted steps
    torch.testing.assert_close(cost, torch.full((2,), 0.75))


def test_repr_cost_early_good_final_bad() -> None:
    target = torch.zeros(2, 2, 2)
    traj = torch.zeros(5, 1, 2, 2, 2)  # T_pred=4
    traj[-1] = 1.0
    all_sum = trajectory_repr_cost(traj, target, sum_all_diffs=True)
    last3 = trajectory_repr_cost(
        traj, target, sum_all_diffs=False, sum_last_n=3
    )
    torch.testing.assert_close(all_sum, torch.ones(1))
    torch.testing.assert_close(last3, torch.ones(1))


def test_repr_cost_early_bad_final_good() -> None:
    target = torch.zeros(2, 2, 2)
    traj = torch.zeros(5, 1, 2, 2, 2)
    traj[1] = 1.0  # first predicted step bad
    all_sum = trajectory_repr_cost(traj, target, sum_all_diffs=True)
    last3 = trajectory_repr_cost(
        traj, target, sum_all_diffs=False, sum_last_n=3
    )
    torch.testing.assert_close(all_sum, torch.ones(1))
    torch.testing.assert_close(last3, torch.zeros(1))


def test_repr_cost_default_ignores_loss_coeff_ramp() -> None:
    target = torch.zeros(2, 2, 2)
    traj = torch.zeros(4, 1, 2, 2, 2)
    traj[1] = 2.0  # only first predicted step
    a = trajectory_repr_cost(
        traj,
        target,
        sum_all_diffs=True,
        loss_coeff_first=0.0,
        loss_coeff_last=1.0,
        apply_loss_coeff_ramp=False,
    )
    b = trajectory_repr_cost(
        traj,
        target,
        sum_all_diffs=True,
        loss_coeff_first=0.0,
        loss_coeff_last=1.0,
        apply_loss_coeff_ramp=True,
    )
    # default MPPI path: first-step error still counts
    assert a.item() == pytest.approx(4.0)
    # unused ReprTargetMPCObjective ramp would zero the first coeff
    assert b.item() < a.item()


def test_representation_cost_class() -> None:
    cost = RepresentationCost(sum_all_diffs=True)
    traj = torch.zeros(3, 2, 2, 2, 2)
    target = torch.zeros(2, 2, 2)
    out = cost(traj, target)
    assert out.shape == (2,)


def test_shift_nominal_trajectory() -> None:
    cfg = _cfg()
    ctrl = MPPI(cfg, horizon=4)
    ctrl.change_horizon(4)
    ctrl.U = torch.arange(8, dtype=torch.float32).view(4, 2)
    ctrl.shift_nominal_trajectory()
    expected = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [0.0, 0.0]])
    torch.testing.assert_close(ctrl.U, expected)


def test_prepare_horizon_shifts_then_truncates() -> None:
    pred = Level1Predictor()
    planner = Level1MPPIPlanner(pred, _cfg(num_samples=2))
    h0 = torch.zeros(1, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    planner._ensure_controllers(1, h0.device, h0.dtype)
    planner.ctrls[0].change_horizon(4)
    planner.ctrls[0].U = torch.arange(8, dtype=torch.float32).view(4, 2)
    planner.last_plan_size = 4
    planner._prepare_horizon(3)
    # shift once → [u1,u2,u3,0] then truncate to 3 → [u1,u2,u3]
    expected = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    torch.testing.assert_close(planner.ctrls[0].U, expected)
    assert planner.ctrls[0].T == 3


def test_toy_dynamics_mppi_reduces_cost() -> None:
    cfg = _cfg(num_samples=48, noise_sigma=1.0, lambda_=0.05)
    ctrl = MPPI(cfg, horizon=3)
    ctrl.change_horizon(3)
    ctrl.U.zero_()
    start = torch.zeros(2)
    target = torch.tensor([0.6, 0.0])
    cost_fn = _toy_rollout_cost(start, target)
    init = cost_fn(bound_plan(ctrl.U.unsqueeze(0), 0.0, 1.0)).item()
    step = ctrl.command(cost_fn, shift_nominal_trajectory=False)
    final = cost_fn(bound_plan(step.U.unsqueeze(0), 0.0, 1.0)).item()
    assert init > final
    bound_u = bound_plan(step.U, 0.0, 1.0)
    assert bound_u[:, 0].sum() > 0


def test_hwm_random_weights_smoke_and_no_grad() -> None:
    torch.manual_seed(0)
    predictor = Level1Predictor()
    predictor.train()
    cfg = _cfg(num_samples=8)
    planner = Level1MPPIPlanner(predictor, cfg)
    h0 = torch.randn(1, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    target = torch.randn(1, VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    for p in predictor.parameters():
        p.grad = None
    h0 = h0.clone().requires_grad_(True)
    result = planner.plan(h0, target, planning_horizon=3)
    assert result.action_sequence.shape == (1, 3, 2)
    assert result.predicted_trajectory.shape == (
        4,
        1,
        FUSED_CHANNELS,
        FEATURE_SIZE,
        FEATURE_SIZE,
    )
    assert result.predicted_visual.shape == (
        4,
        1,
        VISUAL_CHANNELS,
        FEATURE_SIZE,
        FEATURE_SIZE,
    )
    assert result.cost.shape == (1,)
    assert torch.isfinite(result.action_sequence).all()
    assert torch.isfinite(result.predicted_trajectory).all()
    assert torch.isfinite(result.cost).all()
    cand = result.diagnostics["per_env"][0]["candidate_actions"]
    assert cand.shape == (8, 3, 2)
    # no-grad: planning must not populate predictor or input grads
    assert h0.grad is None
    assert all(p.grad is None for p in predictor.parameters())
    assert predictor.training is True
    init_c = result.diagnostics["initial_exec_cost"]
    opt_c = result.diagnostics["optimized_exec_cost"]
    assert init_c.shape == (1,)
    assert opt_c.shape == (1,)


def test_hwm_batch_b2_small() -> None:
    torch.manual_seed(1)
    predictor = Level1Predictor()
    cfg = _cfg(num_samples=4)
    planner = Level1MPPIPlanner(predictor, cfg)
    h0 = torch.randn(2, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    target = torch.randn(2, VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    result = planner.plan(h0, target, planning_horizon=2)
    assert result.action_sequence.shape == (2, 2, 2)
    assert result.predicted_trajectory.shape == (3, 2, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    assert result.cost.shape == (2,)


def test_rollout_candidates_layout() -> None:
    torch.manual_seed(0)
    pred = Level1Predictor()
    pred.eval()
    h0 = torch.randn(2, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    actions = torch.randn(3, 4, 2, 2)  # K=3, T=4, B=2
    with torch.no_grad():
        fused = rollout_action_candidates(pred, h0, actions)
    assert fused.shape == (5, 3, 2, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    torch.testing.assert_close(fused[0, 0, 0], h0[0])
    torch.testing.assert_close(fused[0, 2, 1], h0[1])
