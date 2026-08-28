"""High-level value / scoring model ``Q_H``.

    Q_H(h_tau, candidate_subgoal, g*) -> scalar

This is **not** ``f_H``. It does not predict the next high-level state.
Director's high-level dynamics remain the implicit composition

    f_H^Director = (f_L, pi_L)^K

``Q_H`` only scores a candidate local x/y so that ``pi_H`` can pick among
offline-supported subgoals. The same scorer is intended for later HWM
manager selection; HWM still trains its own explicit ``f_H`` separately.

Input concatenation (33-D):

    h_tau              29-D
    candidate_subgoal   2-D
    final_goal          2-D

Default MLP: ``33 -> 256 -> 256 -> 1``.
"""

from __future__ import annotations

import torch
from torch import nn

from hwm_director.data.state import ACHIEVED_GOAL_DIM, GOAL_DIM, STATE_DIM

VALUE_INPUT_DIM = STATE_DIM + ACHIEVED_GOAL_DIM + GOAL_DIM  # 33


class HighLevelValueModel(nn.Module):
    """Scalar scorer ``Q_H(h_tau, g_tau, g*)``. Not a dynamics model."""

    is_high_level_dynamics = False

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        subgoal_dim: int = ACHIEVED_GOAL_DIM,
        goal_dim: int = GOAL_DIM,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.subgoal_dim = int(subgoal_dim)
        self.goal_dim = int(goal_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.input_dim = self.state_dim + self.subgoal_dim + self.goal_dim
        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        subgoal: torch.Tensor,
        final_goal: torch.Tensor,
    ) -> torch.Tensor:
        """Return a scalar value per row, shape ``(...,)``."""
        x = torch.cat([state, subgoal, final_goal], dim=-1)
        return self.net(x).squeeze(-1)
