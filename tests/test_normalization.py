"""Tests for StateNormalizer. Fail until TODOs in normalization.py are filled."""

from __future__ import annotations

import numpy as np

from hwm_director.data.normalization import StateNormalizer


def test_normalize_preserves_batch_shape() -> None:
    rng = np.random.default_rng(0)
    states = rng.normal(size=(16, 107))
    normalizer = StateNormalizer(eps=1e-8).fit(states)
    out = normalizer.normalize(states)
    assert out.shape == (16, 107)


def test_normalize_preserves_vector_shape() -> None:
    rng = np.random.default_rng(1)
    states = rng.normal(size=(8, 107))
    vector = states[0]
    normalizer = StateNormalizer().fit(states)
    out = normalizer.normalize(vector)
    assert out.shape == (107,)


def test_normalizer_round_trip() -> None:
    rng = np.random.default_rng(2)
    states = rng.normal(loc=3.0, scale=2.0, size=(32, 107))
    states[:, 0] = 5.0
    normalizer = StateNormalizer(eps=1e-8).fit(states)
    recovered = normalizer.denormalize(normalizer.normalize(states))
    np.testing.assert_allclose(recovered, states, rtol=1e-5, atol=1e-6)
