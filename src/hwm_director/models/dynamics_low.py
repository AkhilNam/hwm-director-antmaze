"""One-step low-level world model ``f_L``.

Unified hierarchy only requires **one-step** primitive
dynamics:

    hat{s}_{t+1} = f_L(s_t, a_t)

It does not specify residual vs absolute prediction. This module uses
**delta prediction** as our implementation choice:

    hat{delta}_t = MLP([s_t, a_t])
    hat{s}_{t+1} = s_t + hat{delta}_t

where ``delta_s = s_{t+1} - s_t``. Nearby AntMaze states differ by a small
residual (especially x/y), so learning the change is usually easier than
regurgitating absolute coordinates.

``E`` is still identity: ``s_t`` is the 107-D baseline vector, not a learned
latent. RSSM/JEPA are not used here. ``f_L`` is shared later by Director and
HWM; this file is not either system.
"""

from __future__ import annotations

import torch
from torch import nn

from hwm_director.data.state import STATE_DIM
from hwm_director.data.transitions import ACTION_DIM


class LowLevelDynamicsModel(nn.Module):
    """MLP that maps ``(state, action)`` to a state delta.

    Parameters
    ----------
    state_dim:
        Default ``107``.
    action_dim:
        Default ``8``.
    hidden_dims:
        Hidden layer widths, e.g. ``(256, 256)``.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(hidden_dims)

        layers: list[nn.Module] = []
        in_dim = self.state_dim + self.action_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict ``delta = s_{t+1} - s_t``.

        Parameters
        ----------
        state:
            Shape ``(..., 107)``.
        action:
            Shape ``(..., 8)``.

        Returns
        -------
        torch.Tensor
            Predicted delta, shape ``(..., 107)``.
        """
        concat = torch.cat([state, action], dim=-1)
        return self.net(concat)

    def predict_next_state(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct ``hat{s}_{t+1} = s_t + hat{delta}_t``.

        Shapes match ``forward``: ``(..., 107)``.
        """
        return state + self.forward(state, action)

