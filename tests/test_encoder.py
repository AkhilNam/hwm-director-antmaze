"""Tests for IdentityEncoder."""

from __future__ import annotations

import numpy as np

from hwm_director.models.encoder import IdentityEncoder


def test_identity_encoder_returns_equivalent_copy() -> None:
    encoder = IdentityEncoder()
    state = np.linspace(-1.0, 1.0, 107)
    encoded = encoder.encode(state)
    np.testing.assert_array_equal(encoded, state)
    assert encoded is not state
    encoded[0] = 999.0
    assert state[0] != 999.0
