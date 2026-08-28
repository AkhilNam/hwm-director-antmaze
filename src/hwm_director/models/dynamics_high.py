"""Explicit high-level dynamics ``f_H_phi``.

    h_hat_{tau+1} = f_H_phi(h_tau, g_tau)

This is a separately trained coarse model. It is **not** Director's
implicit composition ``(f_L, pi_L)^K``.

Inputs (raw public API is converted by the HWM wrapper):

    h_tau   29-D high-level / primitive state
    g_tau    2-D local subgoal x/y

Total MLP input: 31-D. Output: 29-D next-state **delta**, same style as
``f_L``:

    delta_H = MLP([h_tau, g_tau])
    h_hat_{tau+1} = h_tau + delta_H

No RSSM, JEPA, or recurrence.

Offline training uses recorded K-step transitions
``(s_t, s_{t+K}[:2]) -> s_{t+K}``. That is supervised on dataset behavior,
not a rollout of the current learned ``pi_L``.
"""

from __future__ import annotations

import torch
from torch import nn

from hwm_director.data.state import ACHIEVED_GOAL_DIM, STATE_DIM

HIGH_LEVEL_DYNAMICS_INPUT_DIM = STATE_DIM + ACHIEVED_GOAL_DIM  # 31


class ExplicitHighLevelDynamics(nn.Module):
    """MLP ``[h_tau, g_tau] -> delta_H`` with next state ``h + delta``.

    Default widths: ``31 -> 256 -> 256 -> 29``.
    """

    is_high_level_dynamics = True

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        subgoal_dim: int = ACHIEVED_GOAL_DIM,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.subgoal_dim = int(subgoal_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.input_dim = self.state_dim + self.subgoal_dim
        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
        """Predict ``delta = h_{tau+1} - h_tau``. Shape ``(..., STATE_DIM)``."""
        return self.net(torch.cat([state, subgoal], dim=-1))

    def predict_next_state(
        self, state: torch.Tensor, subgoal: torch.Tensor
    ) -> torch.Tensor:
        """``h_hat_{tau+1} = h_tau + delta_H``."""
        return state + self.forward(state, subgoal)
