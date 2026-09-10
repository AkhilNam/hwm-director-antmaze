"""Released Level-1 / Level-2 MPPI defaults from original planning YAML.

L1: ``eval_cfg.h_d4rl_planning.level1``
L2: ``eval_cfg.h_d4rl_planning.level2``
Source: ``large_diverse_25maps_l2.yaml`` at SHA ``e197375``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Level1MPPIConfig:
    """Executed L1 MPPI hyperparameters (not textbook defaults)."""

    noise_sigma: float = 5.0  # diagonal *variance* of the sampling covariance
    num_samples: int = 500
    lambda_: float = 0.0025
    z_reg_coeff: float = 0.0
    min_step: float = 0.0
    max_step: float = 1.0
    sum_all_diffs: bool = True
    sum_last_n: int = 3
    loss_coeff_first: float = 1.0
    loss_coeff_last: float = 1.0
    # YAML fields; RunningCost does not apply linspace weighting.
    apply_loss_coeff_ramp: bool = False
    clamp_actions: bool = False
    cost_entity: str = "obs_component"


@dataclass
class Level2MPPIConfig:
    """Executed L2 MPPI hyperparameters (``pi_H``).

    ``clamp_actions`` is forced True in original construction because
    ``latent_actions = l2 and z_dim > 0``. That selects **per-component
    clamp**, not the L1 Euclidean-norm rescale.

    ``z_reg_coeff=0.1`` is stored and ``z_reg`` is computed, but original
    ``_compute_rollout_costs`` never adds it to ``cost_total``.
    ``apply_z_reg`` stays False to match that executed path.

    YAML ``max_plan_length=18`` is overwritten at eval by difficulty
    ``max_plan_length_l2`` (medium 35, hard 47). ``probe_depth=false``:
    ``determine_optimal_depths`` is dormant.
    """

    noise_sigma: float = 10.0  # diagonal *variance*
    num_samples: int = 2000
    lambda_: float = 0.0025
    z_reg_coeff: float = 0.1
    apply_z_reg: bool = False
    min_step: float = -2.5
    max_step: float = 2.5
    sum_all_diffs: bool = True
    sum_last_n: int = 3
    clamp_actions: bool = True
    cost_entity: str = "obs_component"
    min_plan_length: int = 3
    max_plan_length: int = 18
    probe_depth: bool = False
    depth_probe_threshold: float = 25.0
    # Optional dataset percentile bounds [8]. When set, original clamps to
    # ``(min+0.1, max-0.1)`` and ignores YAML ±2.5 / the Ant 8-dim hack.
    z_min_bounds: torch.Tensor | None = None
    z_max_bounds: torch.Tensor | None = None
    z_bound_margin: float = 0.1


@dataclass
class HierarchicalPlannerConfig:
    """Handoff + final-transition constants. No env loop."""

    l2_step_skip: int = 10
    final_trans_steps: int = 15
    mock_l1: bool = False
    replan_every: int = 4
