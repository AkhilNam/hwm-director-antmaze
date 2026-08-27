"""Group one-step transitions into temporally ordered episodes."""

from __future__ import annotations

from typing import Sequence

from hwm_director.data.transitions import Transition


def group_by_episode(
    transitions: Sequence[Transition],
) -> dict[int, list[Transition]]:
    """Map ``episode_id -> transitions`` in collection (temporal) order.

    Each transition appears in exactly one list. Order within an episode is
    the order those steps appear in ``transitions`` (already chronological
    if they came from the Minari loader or the random collector).
    """
    grouped: dict[int, list[Transition]] = {}
    for transition in transitions:
        grouped.setdefault(int(transition.episode_id), []).append(transition)
    return grouped
