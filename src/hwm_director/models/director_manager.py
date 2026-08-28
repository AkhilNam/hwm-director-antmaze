"""Director high-level manager ``pi_H(h_tau, g*) -> g_tau``.

First baseline: a deterministic MLP that predicts a **relative** x/y
displacement from the current torso position. Absolute maze coordinates
are not passed through tanh.

    delta = MLP([h_tau, g*])
    g_tau = h_tau[:2] + delta

This is behavior-cloned from offline trajectories (see
``manager_dataset``). The unified framework later allows offline RL /
behavior-regularized actor-critic for ``pi_H``; that is not implemented
here.
"""

from __future__ import annotations

import torch
from torch import nn

from hwm_director.data.state import ACHIEVED_GOAL_DIM, GOAL_DIM, STATE_DIM

DEFAULT_MAX_SUBGOAL_DISTANCE = 2.0


def clamp_xy_displacement(
    current_xy: torch.Tensor,
    target_xy: torch.Tensor,
    max_distance: float,
) -> torch.Tensor:
    """Limit ``target_xy - current_xy`` to length ``max_distance`` (meters)."""
    delta = target_xy - current_xy
    dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_distance) / dist.clamp(min=1e-8), max=1.0)
    return current_xy + delta * scale


class DirectorManager(nn.Module):
    """MLP ``[state, final_goal] -> relative xy``, then ``g_tau = xy + delta``.

    Default widths: ``(STATE_DIM + GOAL_DIM) -> 256 -> 256 -> 2``.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        goal_dim: int = GOAL_DIM,
        subgoal_dim: int = ACHIEVED_GOAL_DIM,
        hidden_dims: tuple[int, ...] = (256, 256),
        max_subgoal_distance: float = DEFAULT_MAX_SUBGOAL_DISTANCE,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.goal_dim = int(goal_dim)
        self.subgoal_dim = int(subgoal_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.max_subgoal_distance = float(max_subgoal_distance)

        layers: list[nn.Module] = []
        in_dim = self.state_dim + self.goal_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.subgoal_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        final_goal: torch.Tensor,
        *,
        clamp: bool = True,
    ) -> torch.Tensor:
        """Predict a local subgoal in the **same coordinate system as ``state``**.

        Parameters
        ----------
        state:
            Shape ``(..., STATE_DIM)``. First two dims are current x/y.
        final_goal:
            Shape ``(..., GOAL_DIM)``.
        clamp:
            If True, cap displacement length at ``max_subgoal_distance``
            (interpreted in the same units as ``state[:2]``).

        Returns
        -------
        torch.Tensor
            Shape ``(..., 2)``.
        """
        delta = self.net(torch.cat([state, final_goal], dim=-1))
        current_xy = state[..., : self.subgoal_dim]
        proposed = current_xy + delta
        if not clamp:
            return proposed
        return clamp_xy_displacement(
            current_xy, proposed, self.max_subgoal_distance
        )
