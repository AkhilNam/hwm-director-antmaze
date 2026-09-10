"""Released Level-1 MPPI defaults from original planning YAML.

Source: ``large_diverse_25maps_l2.yaml`` ``eval_cfg.d4rl_planning.level1``.
"""

from __future__ import annotations

from dataclasses import dataclass


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
