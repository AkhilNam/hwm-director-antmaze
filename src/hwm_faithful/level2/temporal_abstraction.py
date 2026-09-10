"""Level-2 temporal abstraction: skip-10 states and 10-step action chunks.

Original sources (SHA ``e197375``):

- ``pldm_envs/diverse_maze/d4rl.py`` ``D4RLDataset.__getitem__``
- ``large_diverse_25maps_l2.yaml``: ``l2_step_skip=10``, ``l2_n_steps=6``,
  ``chunked_actions=true``, ``stack_states=1``

Executed window (from a start index):

    l2_n_steps_total = l2_n_steps * l2_step_skip + 1 = 61

    states / proprio:  ``[start : start+61 : 10]``
        → L1 times 0, 10, 20, 30, 40, 50, 60 relative to the window
        → 7 tensors

    primitive actions: ``actions[start : start+60]``  (**no skip**)
        → 60 steps, then ``split(10)`` → 6 chunks
        [0:10], [10:20], [20:30], [30:40], [40:50], [50:60]

Alignment: chunk ``i`` connects state ``i`` (time ``i*skip``) to state
``i+1`` (time ``(i+1)*skip``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from hwm_faithful.shapes import ACTION_DIM, L2_N_STEPS, L2_STEP_SKIP


@dataclass(frozen=True)
class Level2TemporalConfig:
    """Released Diverse Maze L2 window."""

    step_skip: int = L2_STEP_SKIP
    n_steps: int = L2_N_STEPS  # number of L2 transitions, not states

    def __post_init__(self) -> None:
        if self.step_skip < 1:
            raise ValueError(f"step_skip must be >= 1, got {self.step_skip}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")

    @property
    def n_states(self) -> int:
        return self.n_steps + 1

    @property
    def primitive_horizon(self) -> int:
        return self.n_steps * self.step_skip

    @property
    def l1_window(self) -> int:
        return self.primitive_horizon + 1

    @property
    def state_indices(self) -> tuple[int, ...]:
        return tuple(i * self.step_skip for i in range(self.n_states))


class Level2Inputs(NamedTuple):
    """Time-leading L2 tensors matching original after ``transpose(0, 1)``.

    ``states``: ``[T2+1, B, ...]`` subsampled L1 fused (or any leading-time tensor).
    ``action_chunks``: ``[T2, B, skip, 2]``
    ``flat_action_chunks``: ``[T2, B, skip*2]`` C-order flatten of the last two dims.
    ``state_indices``: L1 times of the sampled states, relative to the window.
    """

    states: torch.Tensor
    action_chunks: torch.Tensor
    flat_action_chunks: torch.Tensor
    state_indices: tuple[int, ...]
    chunk_ranges: tuple[tuple[int, int], ...]


def flatten_action_chunk(chunk: torch.Tensor) -> torch.Tensor:
    """Match original ``actions.view(B, -1)`` on ``[B, 10, 2]``.

    C-order: ``[a_0x, a_0y, a_1x, a_1y, ..., a_9x, a_9y]``.
    Accepts ``[..., 10, 2]`` → ``[..., 20]``.
    """
    if chunk.ndim < 2 or chunk.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"action chunk must end with [..., {ACTION_DIM}], got {tuple(chunk.shape)}"
        )
    return chunk.reshape(*chunk.shape[:-2], -1)


def subsample_level2_states(
    states: torch.Tensor,
    config: Level2TemporalConfig | None = None,
) -> torch.Tensor:
    """Take the original L2 window prefix and stride-sample.

    ``states`` is time-leading ``[T, B, ...]``. Returns ``[n_steps+1, B, ...]``.
    """
    cfg = config or Level2TemporalConfig()
    if states.ndim < 1:
        raise ValueError("states must be time-leading")
    need = cfg.l1_window
    if states.shape[0] < need:
        raise ValueError(
            f"need at least {need} L1 states for skip={cfg.step_skip} "
            f"n_steps={cfg.n_steps}, got T={states.shape[0]}"
        )
    window = states[:need]
    sampled = window[:: cfg.step_skip]
    if sampled.shape[0] != cfg.n_states:
        raise RuntimeError(
            f"expected {cfg.n_states} L2 states, got {sampled.shape[0]}"
        )
    return sampled


def chunk_level2_actions(
    actions: torch.Tensor,
    config: Level2TemporalConfig | None = None,
) -> torch.Tensor:
    """Split the primitive-action prefix into skip-length chunks.

    ``actions`` is ``[T_act, B, 2]`` (no skip in the original loader).
    Returns ``[n_steps, B, skip, 2]``.
    """
    cfg = config or Level2TemporalConfig()
    if actions.ndim != 3 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"actions must be [T, B, {ACTION_DIM}], got {tuple(actions.shape)}"
        )
    need = cfg.primitive_horizon
    if actions.shape[0] < need:
        raise ValueError(
            f"need at least {need} primitive actions for skip={cfg.step_skip} "
            f"n_steps={cfg.n_steps}, got T={actions.shape[0]}"
        )
    window = actions[:need]
    n_steps, skip, batch = cfg.n_steps, cfg.step_skip, window.shape[1]
    # [T, B, 2] → [n_steps, skip, B, 2] → [n_steps, B, skip, 2]
    return window.reshape(n_steps, skip, batch, ACTION_DIM).permute(0, 2, 1, 3).contiguous()


def chunk_ranges(config: Level2TemporalConfig | None = None) -> tuple[tuple[int, int], ...]:
    cfg = config or Level2TemporalConfig()
    return tuple(
        (i * cfg.step_skip, (i + 1) * cfg.step_skip) for i in range(cfg.n_steps)
    )


def build_level2_inputs(
    states: torch.Tensor,
    actions: torch.Tensor,
    config: Level2TemporalConfig | None = None,
) -> Level2Inputs:
    """Original-equivalent L2 states + action chunks from a dense L1 window."""
    cfg = config or Level2TemporalConfig()
    sampled = subsample_level2_states(states, cfg)
    chunks = chunk_level2_actions(actions, cfg)
    if sampled.shape[1] != chunks.shape[1]:
        raise ValueError(
            f"state batch {sampled.shape[1]} != action batch {chunks.shape[1]}"
        )
    return Level2Inputs(
        states=sampled,
        action_chunks=chunks,
        flat_action_chunks=flatten_action_chunk(chunks),
        state_indices=cfg.state_indices,
        chunk_ranges=chunk_ranges(cfg),
    )
