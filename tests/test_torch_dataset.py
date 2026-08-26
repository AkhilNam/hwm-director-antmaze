"""Tests for TransitionDataset. Fail until __getitem__ TODOs are filled."""

from __future__ import annotations

import numpy as np
import torch

from hwm_director.data.torch_dataset import TransitionDataset
from hwm_director.data.transitions import Transition


def _one_transition() -> Transition:
    return Transition(
        state=np.zeros(107, dtype=np.float64),
        action=np.zeros(8, dtype=np.float32),
        next_state=np.ones(107, dtype=np.float64),
        goal=np.zeros(2, dtype=np.float64),
    )


def test_dataset_length() -> None:
    dataset = TransitionDataset([_one_transition(), _one_transition()])
    assert len(dataset) == 2


def test_dataset_item_tensor_shapes() -> None:
    dataset = TransitionDataset([_one_transition()])
    item = dataset[0]
    assert item["state"].shape == (107,)
    assert item["action"].shape == (8,)
    assert item["next_state"].shape == (107,)
    assert item["state"].dtype == torch.float32
    assert "goal" not in item
