"""Level-2 MPPI planner: original ``pi_H``.

    z_{0:T2-1} = pi_H(H_t, o*; f_H)

Planning samples ``z`` directly. The train-time action posterior is not used.

Original mapping (SHA ``e197375``):

- ``MPPIPlanner(..., l2=True)`` with ``latent_actions=True``
- ``LearnedDynamics`` passes candidates as ``latents=`` into ``f_H``
- ``RunningCost`` on ``obs_component`` (visual 16 channels)
- ``normalize_actions(..., clamp_actions=True)``: per-component clamp
- ``z_reg`` is computed then dropped
- one ``command(shift_nominal_trajectory=False)``
- returned trajectory is rolled from **unbounded** ``U``
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch

from hwm_faithful.level2.predictor import Level2Predictor
from hwm_faithful.models.conv_predictor import split_fused
from hwm_faithful.planning.actions import bound_z
from hwm_faithful.planning.config import Level2MPPIConfig
from hwm_faithful.planning.costs import RepresentationCost, validate_visual, z_regularization
from hwm_faithful.planning.mppi import MPPI
from hwm_faithful.shapes import (
    FEATURE_SIZE,
    FUSED_CHANNELS,
    VISUAL_CHANNELS,
    Z_DIM,
)


class Level2PlanResult(NamedTuple):
    z_sequence: torch.Tensor  # [B, T2, 8] bound macros
    predicted_H: torch.Tensor  # fused [T2+1, B, 18, 43, 43]
    predicted_visual: torch.Tensor  # [T2+1, B, 16, 43, 43]
    cost: torch.Tensor  # [B] visual running cost of the returned plan
    diagnostics: dict[str, Any]


def rollout_z_candidates(
    predictor: Level2Predictor,
    h0: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Batch-K Level-2 rollouts without a Python loop over samples.

    ``h0``: ``[B, 18, 43, 43]``
    ``z``: ``[K, T2, B, 8]``
    returns fused ``[T2+1, K, B, 18, 43, 43]``
    """
    if h0.ndim != 4:
        raise ValueError(f"h0 must be [B, 18, 43, 43], got {tuple(h0.shape)}")
    if z.ndim != 4 or z.shape[-1] != Z_DIM:
        raise ValueError(f"z must be [K, T2, B, {Z_DIM}], got {tuple(z.shape)}")
    k, t, b, _ = z.shape
    if h0.shape[0] != b:
        raise ValueError(f"h0 batch {h0.shape[0]} != z B={b}")
    h0_flat = h0.repeat(k, 1, 1, 1)
    z_flat = z.permute(1, 0, 2, 3).contiguous().reshape(t, k * b, Z_DIM)
    rolled = predictor.rollout(h0_flat, z_flat)
    return rolled.fused.view(t + 1, k, b, *rolled.fused.shape[2:])


def l2_env_plan_size(
    env_step: int,
    *,
    max_plan_length: int,
    min_plan_length: int = 3,
    step_skip: int = 10,
) -> int:
    """Env-loop T2 from original ``_perform_mpc`` (bilevel).

    ``plan_size = ceil((T2_max * skip - i) / skip)``, floored at
    ``min_plan_length``. ``probe_depth`` is not used.
    """
    if env_step < 0:
        raise ValueError(f"env_step must be >= 0, got {env_step}")
    max_horizon_l1 = max_plan_length * step_skip
    plan_size = (max_horizon_l1 - env_step + step_skip - 1) // step_skip
    return max(int(plan_size), int(min_plan_length))


class Level2MPPIPlanner:
    """Per-env MPPI over ``f_H``. Posterior is not an input and is not called."""

    def __init__(
        self,
        predictor: Level2Predictor,
        config: Level2MPPIConfig | None = None,
        *,
        n_envs: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.predictor = predictor
        self.config = config or Level2MPPIConfig()
        self.n_envs = n_envs
        self.dtype = dtype
        self.cost_fn = RepresentationCost(
            sum_all_diffs=self.config.sum_all_diffs,
            sum_last_n=self.config.sum_last_n,
        )
        self.ctrls: list[MPPI] = []
        self.target_visual: torch.Tensor | None = None
        self.last_plan_size: int | None = None

    def set_target(self, target_visual: torch.Tensor) -> None:
        if target_visual.ndim == 3:
            target_visual = target_visual.unsqueeze(0)
        validate_visual(target_visual, "target_visual")
        self.target_visual = target_visual.detach()

    def reset(self) -> None:
        for ctrl in self.ctrls:
            ctrl.reset()
        self.last_plan_size = None

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        return bound_z(
            z,
            min_step=self.config.min_step,
            max_step=self.config.max_step,
            per_dim_min=self.config.z_min_bounds,
            per_dim_max=self.config.z_max_bounds,
            margin=self.config.z_bound_margin,
        )

    def _prepare_horizon(self, plan_size: int) -> None:
        if self.last_plan_size is not None and plan_size < self.last_plan_size:
            n_shift = self.last_plan_size - plan_size
            for ctrl in self.ctrls:
                for _ in range(n_shift):
                    ctrl.shift_nominal_trajectory()
        for ctrl in self.ctrls:
            ctrl.change_horizon(plan_size)

    def shift_nominal_trajectory(self) -> None:
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
                    action_dim=Z_DIM,
                    device=device,
                    dtype=dtype,
                    bound_fn=self._bound,
                )
                for _ in range(batch_size)
            ]
            self.last_plan_size = None
        self.n_envs = batch_size

    def _executable_cost(
        self,
        h0_i: torch.Tensor,
        U: torch.Tensor,
        target_i: torch.Tensor,
    ) -> torch.Tensor:
        bound_u = self._bound(U).unsqueeze(0)
        fused = rollout_z_candidates(
            self.predictor,
            h0_i.unsqueeze(0),
            bound_u.unsqueeze(2),
        )[:, 0, 0]
        obs = split_fused(fused.unsqueeze(1))[0]
        return self.cost_fn(obs, target_i).reshape(())

    def _candidate_rollout_cost(
        self,
        h0_i: torch.Tensor,
        perturbed: torch.Tensor,
        target_i: torch.Tensor,
    ) -> torch.Tensor:
        fused = rollout_z_candidates(
            self.predictor,
            h0_i.unsqueeze(0),
            perturbed.unsqueeze(2),
        )[:, :, 0]
        obs, _ = split_fused(fused)
        rollout = self.cost_fn(obs, target_i)
        if self.config.apply_z_reg:
            return rollout + z_regularization(perturbed, self.config.z_reg_coeff)
        return rollout

    @torch.no_grad()
    def plan(
        self,
        current_state: torch.Tensor,
        target_visual: torch.Tensor | None = None,
        plan_horizon: int = 3,
    ) -> Level2PlanResult:
        if current_state.ndim != 4 or current_state.shape[1:] != (
            FUSED_CHANNELS,
            FEATURE_SIZE,
            FEATURE_SIZE,
        ):
            raise ValueError(
                f"current_state must be [B, {FUSED_CHANNELS}, {FEATURE_SIZE}, "
                f"{FEATURE_SIZE}], got {tuple(current_state.shape)}"
            )
        if plan_horizon < 1:
            raise ValueError(f"plan_horizon must be >= 1, got {plan_horizon}")
        if target_visual is not None:
            self.set_target(target_visual)
        if self.target_visual is None:
            raise ValueError("target visual representation is required")

        h0 = current_state.detach()
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
            plan_size = int(plan_horizon)
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
                z_reg = z_regularization(step.perturbed_actions, self.config.z_reg_coeff)
                env_diagnostics.append(
                    {
                        "weights": step.weights,
                        "sampled_costs": step.costs,
                        "rollout_costs": step.rollout_costs,
                        "perturbation_costs": step.perturbation_costs,
                        "candidate_z": step.perturbed_actions,
                        "z_reg": z_reg,
                        "z_reg_applied": self.config.apply_z_reg,
                    }
                )

            raw_z = torch.stack(raw_plans, dim=0)
            fused = self.predictor.rollout(
                h0, raw_z.permute(1, 0, 2).contiguous()
            ).fused
            obs, _proprio = split_fused(fused)
            bound_macros = self._bound(raw_z)
            self.last_plan_size = plan_size
            diagnostics: dict[str, Any] = {
                "n_refinement_iterations": 1,
                "shift_nominal_in_plan": False,
                "candidate_layout": "[K, T2, 8] per env; model sees [T2, K, 8]",
                "returned_z_layout": "[B, T2, 8]",
                "cost_channels": VISUAL_CHANNELS,
                "posterior_used": False,
                "z_reg_applied": self.config.apply_z_reg,
                "initial_exec_cost": torch.stack(exec_costs_init),
                "optimized_exec_cost": torch.stack(exec_costs_opt),
                "raw_U": raw_z,
                "bound_U": bound_macros,
                "per_env": env_diagnostics,
            }
            return Level2PlanResult(
                z_sequence=bound_macros,
                predicted_H=fused,
                predicted_visual=obs,
                cost=torch.stack(exec_costs_opt),
                diagnostics=diagnostics,
            )
        finally:
            self.predictor.train(orig_training)
