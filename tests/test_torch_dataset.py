"""Tests for TransitionDataset."""

from __future__ import annotations

import numpy as np
import torch

from hwm_director.data.state import STATE_DIM
from hwm_director.data.torch_dataset import TransitionDataset
from hwm_director.data.transitions import ACTION_DIM
from tests.helpers import make_transition


def test_dataset_length() -> None:
    dataset = TransitionDataset([make_transition(), make_transition()])
    assert len(dataset) == 2


def test_dataset_item_tensor_shapes() -> None:
    dataset = TransitionDataset(
        [make_transition(next_state=np.ones(STATE_DIM, dtype=np.float64))]
    )
    item = dataset[0]
    assert item["state"].shape == (STATE_DIM,)
    assert item["action"].shape == (ACTION_DIM,)
    assert item["next_state"].shape == (STATE_DIM,)
    assert item["state"].dtype == torch.float32
    assert "goal" not in item
