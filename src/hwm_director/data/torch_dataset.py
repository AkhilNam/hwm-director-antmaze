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

    - ``state``: ``(STATE_DIM,)``
    - ``action``: ``(ACTION_DIM,)``
    - ``next_state``: ``(STATE_DIM,)``
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

    @classmethod
    def from_arrays(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
    ) -> TransitionDataset:
        """Build from stacked ``(N, ...)`` arrays (avoids copying 1M objects)."""
        dataset = cls.__new__(cls)
        dataset.transitions = []
        dataset.states = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        dataset.actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
        dataset.next_states = torch.as_tensor(
            np.asarray(next_states), dtype=torch.float32
        )
        return dataset

    def __len__(self) -> int:
        return int(self.states.shape[0])

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
            ``(STATE_DIM,)``, ``(ACTION_DIM,)``, ``(STATE_DIM,)``.
        """
        return {
            "state": self.states[index],
            "action": self.actions[index],
            "next_state": self.next_states[index],
        }
