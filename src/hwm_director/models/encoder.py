"""Identity encoder: E(s) = s.

This is not a learned encoder. Returning a copy keeps later in-place edits
from mutating stored transitions.
"""

from __future__ import annotations

import numpy as np


class IdentityEncoder:
    """Pass-through encoder ``E(s) = s`` (NumPy only)."""

    def encode(self, state: np.ndarray) -> np.ndarray:
        """Return a copy of ``state`` with the same values."""
        return np.asarray(state).copy()
