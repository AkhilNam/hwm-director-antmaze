"""Tests for LowLevelDynamicsModel."""

from __future__ import annotations

import torch

from hwm_director.models.dynamics_low import LowLevelDynamicsModel


def test_forward_delta_shape_batched() -> None:
    model = LowLevelDynamicsModel(hidden_dims=(32, 32))
    state = torch.zeros(4, 107)
    action = torch.zeros(4, 8)
    delta = model(state, action)
    assert delta.shape == (4, 107)


def test_predict_next_state_shape_unbatched() -> None:
    model = LowLevelDynamicsModel(hidden_dims=(16,))
    state = torch.zeros(107)
    action = torch.zeros(8)
    next_state = model.predict_next_state(state, action)
    assert next_state.shape == (107,)
