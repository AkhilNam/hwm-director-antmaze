"""Tests for LowLevelDynamicsModel."""

from __future__ import annotations

import torch

from hwm_director.data.state import STATE_DIM
from hwm_director.data.transitions import ACTION_DIM
from hwm_director.models.dynamics_low import LowLevelDynamicsModel


def test_forward_delta_shape_batched() -> None:
    model = LowLevelDynamicsModel(hidden_dims=(32, 32))
    state = torch.zeros(4, STATE_DIM)
    action = torch.zeros(4, ACTION_DIM)
    delta = model(state, action)
    assert delta.shape == (4, STATE_DIM)


def test_predict_next_state_shape_unbatched() -> None:
    model = LowLevelDynamicsModel(hidden_dims=(16,))
    state = torch.zeros(STATE_DIM)
    action = torch.zeros(ACTION_DIM)
    next_state = model.predict_next_state(state, action)
    assert next_state.shape == (STATE_DIM,)
