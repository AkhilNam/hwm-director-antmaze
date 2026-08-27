"""Per-dimension mean/std normalizer for baseline states of shape ``(STATE_DIM,)``.

Fit on a stacked array of shape ``(N, STATE_DIM)``. ``normalize`` /
``denormalize`` accept ``(N, STATE_DIM)`` or a single ``(STATE_DIM,)`` and
must preserve that shape.

``std`` is guarded with ``eps`` so near-constant dimensions do not blow up.
The map should be invertible:

    denormalize(normalize(x)) ≈ x
"""

from __future__ import annotations

import numpy as np

from hwm_director.data.state import STATE_DIM


class StateNormalizer:
    """Standardize each state dimension independently."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, states: np.ndarray) -> StateNormalizer:
        """Estimate per-dimension mean and std from training states.

        Parameters
        ----------
        states:
            ``(N, STATE_DIM)`` or ``(STATE_DIM,)``.

        Returns
        -------
        self
        """
        states = np.reshape(states, (-1, STATE_DIM))
        self.mean = np.mean(states, axis=0)
        self.std = np.std(states, axis=0)
        self.std = np.maximum(self.std, self.eps)
        return self

    def normalize(self, states: np.ndarray) -> np.ndarray:
        """Return ``(states - mean) / std`` with the same shape as ``states``."""
        if self.mean is None or self.std is None:
            raise ValueError("fit() must be called before normalize()")
        broadcast_mean = np.broadcast_to(self.mean, states.shape)
        broadcast_std = np.broadcast_to(self.std, states.shape)
        return (states - broadcast_mean) / broadcast_std

    def denormalize(self, states: np.ndarray) -> np.ndarray:
        """Inverse of ``normalize``: ``states * std + mean``."""
        if self.mean is None or self.std is None:
            raise ValueError("fit() must be called before denormalize()")

        return states * self.std + self.mean
