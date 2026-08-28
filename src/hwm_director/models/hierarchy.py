"""Shared hierarchical interface ``M_hier = (pi_H, f_H, pi_L, f_L)``.

Director and HWM both expose this outer API.

Director ``f_H`` is the implicit K-step composition ``(f_L, pi_L)^K``.
HWM ``f_H`` is a separately trained ``f_H_phi``. Both use the same
``pi_L`` and ``f_L``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class HierarchicalController(Protocol):
    """Outer API shared by Director and HWM."""

    horizon_k: int

    def select_high_level_command(
        self, state: np.ndarray, final_goal: np.ndarray
    ) -> np.ndarray:
        """``pi_H``: map ``(h_tau, g*)`` to a local subgoal ``g_tau``."""

    def high_level_transition(
        self, state: np.ndarray, subgoal: np.ndarray
    ) -> np.ndarray:
        """``f_H``: predict the state after one high-level interval of length ``K``.

        Director: ``K`` closed-loop ``(pi_L, f_L)`` steps.
        HWM: one forward of explicit ``f_H_phi``.
        """

    def low_level_action(
        self, state: np.ndarray, subgoal: np.ndarray
    ) -> np.ndarray:
        """``pi_L``: map ``(s_t, g_tau)`` to a primitive action."""
