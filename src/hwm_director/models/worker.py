"""Goal-conditioned low-level worker ``pi_L``.

    a_t = pi_L(s_t, g_tau)

First baseline: ``g_tau`` is a target torso x/y in R^2, and the worker emits
an 8-D Ant torque in [-1, 1] (tanh). This is a deterministic MLP, not a
sampled distribution.

This module is **not** Director and **not** HWM. Both of those systems will
call the same ``pi_L`` once a high-level command has been turned into ``g_tau``.
There is no ``pi_H`` / ``f_H`` here. RSSM and JEPA are still unused; ``s_t``
is the identity-encoded 107-D state.
"""

from __future__ import annotations

import torch
from torch import nn

from hwm_director.data.state import ACHIEVED_GOAL_DIM, STATE_DIM
from hwm_director.data.transitions import ACTION_DIM


class GoalConditionedWorker(nn.Module):
    """MLP ``[state, subgoal] -> action`` with tanh squashing.

    Default widths: ``109 -> 256 -> 256 -> 8``.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        subgoal_dim: int = ACHIEVED_GOAL_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.subgoal_dim = int(subgoal_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(hidden_dims)

        layers: list[nn.Module] = []
        in_dim = self.state_dim + self.subgoal_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
        """Predict an action in ``[-1, 1]``.

        Parameters
        ----------
        state:
            Shape ``(..., 107)``.
        subgoal:
            Shape ``(..., 2)``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., 8)``, values in ``[-1, 1]`` after tanh.
        """
        return torch.tanh(self.net(torch.cat([state, subgoal], dim=-1)))
