"""PyTorch Dataset of one-step tuples ``(state, action, next_state)``.

``f_L`` does not take the maze task goal, so ``Transition.goal`` is omitted.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from hwm_director.data.transitions import Transition


class TransitionDataset(Dataset):
    """Indexable ``(state, action, next_state)`` tensors.

    Each ``__getitem__`` returns a dict of 1-D float32 tensors:

    - ``state``: ``(107,)``
    - ``action``: ``(8,)``
    - ``next_state``: ``(107,)``
    """

    def __init__(self, transitions: Sequence[Transition]) -> None:
        self.transitions = list(transitions)
        self.states = torch.as_tensor(
            np.stack([t.state for t in self.transitions]), dtype=torch.float32
        )
        self.actions = torch.as_tensor(
            np.stack([t.action for t in self.transitions]), dtype=torch.float32
        )
        self.next_states = torch.as_tensor(
            np.stack([t.next_state for t in self.transitions]), dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one transition as CPU float32 tensors.

        Parameters
        ----------
        index:
            Integer in ``[0, len(self))``.

        Returns
        -------
        dict
            Keys ``state``, ``action``, ``next_state`` with shapes
            ``(107,)``, ``(8,)``, ``(107,)``.
        """
        return {
            "state": self.states[index],
            "action": self.actions[index],
            "next_state": self.next_states[index],
        }
