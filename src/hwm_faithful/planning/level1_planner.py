"""Level-1 MPPI planner: original ``pi_L``.

This is an online planning procedure induced by the learned Level-1 world
model, **not** the BC worker in ``hwm_director``.

    a_{0:T-1} = pi_L(h_t, o*, f_L)

Original mapping (SHA ``e197375``):

- ``pldm/planning/planners/mppi_planner.py``: ``MPPIPlanner.plan``
- ``pldm/planning/planners/mppi_torch.py``: ``MPPI._command``
- ``pldm/planning/planners/mppi_planner.py``: ``RunningCost`` (visual MSE)
- ``pldm/planning/mpc.py``: action-norm bound + dataset unnormalize

``num_refinement_steps`` is stored in the original planner but never looped:
each ``plan()`` call performs **one** ``command(..., shift_nominal_trajectory=False)``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch

from hwm_faithful.models.conv_predictor import Level1Predictor, split_fused
from hwm_faithful.planning.actions import unnormalize_action
from hwm_faithful.planning.config import Level1MPPIConfig
from hwm_faithful.planning.costs import RepresentationCost, validate_visual
from hwm_faithful.planning.mppi import MPPI, bound_plan
from hwm_faithful.shapes import (
    ACTION_DIM,
    FEATURE_SIZE,
    FUSED_CHANNELS,
    VISUAL_CHANNELS,
)


class Level1PlanResult(NamedTuple):
    """Structured L1 plan, analogous to original ``PlanningResult``."""

    action_sequence: torch.Tensor  # [B, T, 2] bound then unnormalized
    predicted_trajectory: torch.Tensor  # fused [T+1, B, 18, 43, 43]
    predicted_visual: torch.Tensor  # [T+1, B, 16, 43, 43]
    cost: torch.Tensor  # [B] visual running cost of the returned plan
    diagnostics: dict[str, Any]


def rollout_action_candidates(
    predictor: Level1Predictor,
    h0: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Batch-K Level-1 rollouts without a Python loop over samples.

    Parameters
    ----------
    h0:
        ``[B, 18, 43, 43]``
    actions:
        ``[K, T, B, 2]``

    Returns
    -------
    fused:
        ``[T+1, K, B, 18, 43, 43]``
    """
    if h0.ndim != 4:
        raise ValueError(f"h0 must be [B, 18, 43, 43], got {tuple(h0.shape)}")
    if actions.ndim != 4 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"actions must be [K, T, B, 2], got {tuple(actions.shape)}")
    k, t, b, _ = actions.shape
    if h0.shape[0] != b:
        raise ValueError(f"h0 batch {h0.shape[0]} != actions B={b}")
    h0_flat = h0.repeat(k, 1, 1, 1)
    act_flat = actions.permute(1, 0, 2, 3).contiguous().reshape(t, k * b, ACTION_DIM)
    rolled = predictor.rollout(h0_flat, act_flat)
    fused = rolled.fused.view(t + 1, k, b, *rolled.fused.shape[2:])
    return fused


class Level1MPPIPlanner:
    """Per-env MPPI over ``f_L``, matching original ``MPPIPlanner`` for L1."""

    def __init__(
        self,
        predictor: Level1Predictor,
        config: Level1MPPIConfig | None = None,
        *,
        n_envs: int | None = None,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.predictor = predictor
        self.config = config or Level1MPPIConfig()
        self.n_envs = n_envs
        self.action_mean = action_mean
        self.action_std = action_std
        self.dtype = dtype
        self.cost_fn = RepresentationCost(
            sum_all_diffs=self.config.sum_all_diffs,
            sum_last_n=self.config.sum_last_n,
            loss_coeff_first=self.config.loss_coeff_first,
            loss_coeff_last=self.config.loss_coeff_last,
            apply_loss_coeff_ramp=self.config.apply_loss_coeff_ramp,
        )
        self.ctrls: list[MPPI] = []
        self.target_visual: torch.Tensor | None = None
        self.last_plan_size: int | None = None

    def set_target(self, target_visual: torch.Tensor) -> None:
        """Store goal visual maps ``[B, 16, 43, 43]`` (original ``obs_component``)."""
        if target_visual.ndim == 3:
            target_visual = target_visual.unsqueeze(0)
        validate_visual(target_visual, "target_visual")
        self.target_visual = target_visual.detach()

    def reset(self) -> None:
        for ctrl in self.ctrls:
            ctrl.reset()
        self.last_plan_size = None

    def _prepare_horizon(self, plan_size: int) -> None:
        """Original ``MPPIPlanner.plan`` size-change shift, then ``change_horizon``."""
        if self.last_plan_size is not None and plan_size < self.last_plan_size:
            n_shift = self.last_plan_size - plan_size
            for ctrl in self.ctrls:
                for _ in range(n_shift):
                    ctrl.shift_nominal_trajectory()
        for ctrl in self.ctrls:
            ctrl.change_horizon(plan_size)

    def shift_nominal_trajectory(self) -> None:
        """Env-level receding-horizon helper. Original ``plan()`` does **not** call this."""
        for ctrl in self.ctrls:
            ctrl.shift_nominal_trajectory()

    def _ensure_controllers(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if self.n_envs is not None and batch_size != self.n_envs:
            raise ValueError(
                f"planner n_envs={self.n_envs} != current batch {batch_size}"
            )
        if len(self.ctrls) != batch_size or (
            self.ctrls and self.ctrls[0].U.device != device
        ):
            self.ctrls = [
                MPPI(
                    self.config,
                    action_dim=ACTION_DIM,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(batch_size)
            ]
            self.last_plan_size = None
        self.n_envs = batch_size

    def _bound(self, actions: torch.Tensor) -> torch.Tensor:
        return bound_plan(
            actions,
            self.config.min_step,
            self.config.max_step,
            clamp_components=self.config.clamp_actions,
        )

    def _executable_cost(
        self,
        h0_i: torch.Tensor,
        U: torch.Tensor,
        target_i: torch.Tensor,
    ) -> torch.Tensor:
        """Visual running cost of a bound nominal plan (what the env would get)."""
        bound_u = self._bound(U).unsqueeze(0)  # [1, T, 2]
        fused = rollout_action_candidates(
            self.predictor,
            h0_i.unsqueeze(0),
            bound_u.unsqueeze(2),  # [1, T, 1, 2]
        )[:, 0, 0]  # [T+1, 18, 43, 43]
        obs = split_fused(fused.unsqueeze(1))[0]  # [T+1, 1, 16, 43, 43]
        return self.cost_fn(obs, target_i).reshape(())

    def _candidate_rollout_cost(
        self,
        h0_i: torch.Tensor,
        perturbed: torch.Tensor,
        target_i: torch.Tensor,
    ) -> torch.Tensor:
        """``perturbed`` is ``[K, T, 2]``. Returns ``[K]``."""
        fused = rollout_action_candidates(
            self.predictor,
            h0_i.unsqueeze(0),
            perturbed.unsqueeze(2),  # [K, T, 1, 2]
        )[:, :, 0]  # [T+1, K, 18, 43, 43]
        obs, _ = split_fused(fused)
        return self.cost_fn(obs, target_i)

    @torch.no_grad()
    def plan(
        self,
        current_fused_state: torch.Tensor,
        target_visual_representation: torch.Tensor | None = None,
        planning_horizon: int = 10,
    ) -> Level1PlanResult:
        """One original-style L1 plan call (single MPPI sample-and-update)."""
        if current_fused_state.ndim != 4:
            raise ValueError(
                f"current_fused_state must be [B, {FUSED_CHANNELS}, "
                f"{FEATURE_SIZE}, {FEATURE_SIZE}], got {tuple(current_fused_state.shape)}"
            )
        if current_fused_state.shape[1:] != (
            FUSED_CHANNELS,
            FEATURE_SIZE,
            FEATURE_SIZE,
        ):
            raise ValueError(
                f"current_fused_state must be [B, {FUSED_CHANNELS}, "
                f"{FEATURE_SIZE}, {FEATURE_SIZE}], got {tuple(current_fused_state.shape)}"
            )
        if planning_horizon < 1:
            raise ValueError(f"planning_horizon must be >= 1, got {planning_horizon}")
        if target_visual_representation is not None:
            self.set_target(target_visual_representation)
        if self.target_visual is None:
            raise ValueError("target visual representation is required")

        h0 = current_fused_state.detach()
        target = self.target_visual.to(device=h0.device, dtype=h0.dtype)
        if target.shape[0] != h0.shape[0]:
            raise ValueError(
                f"target batch {target.shape[0]} != fused batch {h0.shape[0]}"
            )
        validate_visual(target, "target_visual")

        orig_training = self.predictor.training
        self.predictor.train(False)
        try:
            self._ensure_controllers(h0.shape[0], h0.device, h0.dtype)
            plan_size = int(planning_horizon)
            self._prepare_horizon(plan_size)
            raw_plans: list[torch.Tensor] = []
            env_diagnostics: list[dict[str, Any]] = []
            exec_costs_init: list[torch.Tensor] = []
            exec_costs_opt: list[torch.Tensor] = []

            for i, ctrl in enumerate(self.ctrls):
                h0_i = h0[i]
                tgt_i = target[i]
                exec_costs_init.append(self._executable_cost(h0_i, ctrl.U, tgt_i))

                def rollout_cost_fn(
                    perturbed: torch.Tensor,
                    h0_i: torch.Tensor = h0_i,
                    tgt_i: torch.Tensor = tgt_i,
                ) -> torch.Tensor:
                    return self._candidate_rollout_cost(h0_i, perturbed, tgt_i)

                step = ctrl.command(
                    rollout_cost_fn, shift_nominal_trajectory=False
                )
                raw_plans.append(step.U)
                exec_costs_opt.append(self._executable_cost(h0_i, step.U, tgt_i))
                env_diagnostics.append(
                    {
                        "weights": step.weights,
                        "sampled_costs": step.costs,
                        "rollout_costs": step.rollout_costs,
                        "perturbation_costs": step.perturbation_costs,
                        "candidate_actions": step.perturbed_actions,
                    }
                )

            raw_actions = torch.stack(raw_plans, dim=0)  # [B, T, 2]
            # Original rolls the *unbounded* U for the returned trajectory.
            fused = self.predictor.rollout(
                h0, raw_actions.permute(1, 0, 2).contiguous()
            ).fused
            obs, _proprio = split_fused(fused)

            bound_actions = self._bound(raw_actions)
            actions = unnormalize_action(
                bound_actions, mean=self.action_mean, std=self.action_std
            )
            self.last_plan_size = plan_size

            init_cost = torch.stack(exec_costs_init)
            opt_cost = torch.stack(exec_costs_opt)
            diagnostics: dict[str, Any] = {
                "n_refinement_iterations": 1,
                "shift_nominal_in_plan": False,
                "candidate_layout": "[K, T, 2] per env; model sees [T, K, 2]",
                "returned_action_layout": "[B, T, 2]",
                "cost_channels": VISUAL_CHANNELS,
                "loss_coeff_ramp_applied": self.config.apply_loss_coeff_ramp,
                "initial_exec_cost": init_cost,
                "optimized_exec_cost": opt_cost,
                "raw_U": raw_actions,
                "bound_U": bound_actions,
                "per_env": env_diagnostics,
            }
            return Level1PlanResult(
                action_sequence=actions,
                predicted_trajectory=fused,
                predicted_visual=obs,
                cost=opt_cost,
                diagnostics=diagnostics,
            )
        finally:
            self.predictor.train(orig_training)
