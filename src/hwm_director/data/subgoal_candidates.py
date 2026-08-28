"""Data-supported local subgoal candidates for high-level ``pi_H``.

Retrieval looks up offline source states and returns their recorded K-step
future x/y. This is **not** a learned ``f_H``.

Modes
-----
``xy``:
    nearby source x/y (original Director-Value).
``state``:
    nearby source x/y as a coarse maze filter, then nearest **normalized
    29-D** source states.
``hybrid``:
    rank by ``alpha * ||z - z'|| + beta * ||z[:2] - z'[:2]||`` where ``z``
    is the train-only normalized 29-D state.

Director-Value and HWM share this index.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from hwm_director.data.high_level_transitions import HighLevelTransition
from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import STATE_DIM
from hwm_director.data.trajectories import group_by_episode
from hwm_director.data.transitions import Transition
from hwm_director.data.worker_dataset import DEFAULT_HORIZON_K
from hwm_director.models.director_manager import DEFAULT_MAX_SUBGOAL_DISTANCE

DEFAULT_CANDIDATE_RADIUS = 0.75
DEFAULT_N_CANDIDATES = 32
DEFAULT_GRID_CELL = 0.25
DEFAULT_STATE_DISTANCE_WEIGHT = 1.0
DEFAULT_XY_DISTANCE_WEIGHT = 1.0
DEFAULT_SOURCE_DISTANCE_PERCENTILE = 90.0
CANDIDATE_RECORD_KEYS = (
    "future_xy",
    "source_state",
    "source_xy",
    "source_episode_id",
    "source_t",
    "source_xy_distance",
    "source_state_distance",
    "normalized_xy_distance",
    "hybrid_distance",
)


class SubgoalCandidateIndex:
    """Offline ``(source_state, recorded K-step future xy)`` lookup."""

    def __init__(
        self,
        current_xy: np.ndarray,
        future_xy: np.ndarray,
        *,
        source_states: np.ndarray | None = None,
        episode_ids: np.ndarray | None = None,
        times: np.ndarray | None = None,
        cell_size: float = DEFAULT_GRID_CELL,
        normalizer: StateNormalizer | None = None,
    ) -> None:
        self.current_xy = np.asarray(current_xy, dtype=np.float64)
        self.future_xy = np.asarray(future_xy, dtype=np.float64)
        if self.current_xy.shape != self.future_xy.shape:
            raise ValueError("current_xy and future_xy must have the same shape")
        if self.current_xy.ndim != 2 or self.current_xy.shape[1] != 2:
            raise ValueError("expected (N, 2) current_xy")
        n = int(self.current_xy.shape[0])
        if source_states is None:
            source_states = np.zeros((n, STATE_DIM), dtype=np.float64)
            if n:
                source_states[:, :2] = self.current_xy
        self.source_states = np.asarray(source_states, dtype=np.float64)
        if self.source_states.shape != (n, STATE_DIM):
            raise ValueError(
                f"source_states has shape {self.source_states.shape}, "
                f"expected ({n}, {STATE_DIM})"
            )
        self.episode_ids = (
            np.asarray(episode_ids, dtype=np.int64)
            if episode_ids is not None
            else np.full(n, -1, dtype=np.int64)
        )
        self.times = (
            np.asarray(times, dtype=np.int64)
            if times is not None
            else np.arange(n, dtype=np.int64)
        )
        self.cell_size = float(cell_size)
        self.normalizer = normalizer
        self.source_states_n: np.ndarray | None = None
        self._grid: dict[tuple[int, int], np.ndarray] = {}
        self._build_grid()
        if normalizer is not None and n:
            self.set_normalizer(normalizer)

    def set_normalizer(self, normalizer: StateNormalizer) -> None:
        """Cache train-only normalized source states. Does not refit."""
        if normalizer.mean is None or normalizer.std is None:
            raise ValueError("normalizer must be fit before set_normalizer()")
        self.normalizer = normalizer
        if self.source_states.shape[0] == 0:
            self.source_states_n = np.zeros((0, STATE_DIM), dtype=np.float64)
            return
        self.source_states_n = np.asarray(
            normalizer.normalize(self.source_states), dtype=np.float64
        )

    def _cell(self, xy: np.ndarray) -> tuple[int, int]:
        return (
            int(np.floor(xy[0] / self.cell_size)),
            int(np.floor(xy[1] / self.cell_size)),
        )

    def _build_grid(self) -> None:
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, xy in enumerate(self.current_xy):
            buckets[self._cell(xy)].append(i)
        self._grid = {key: np.asarray(idx, dtype=np.int64) for key, idx in buckets.items()}

    @classmethod
    def from_high_level_examples(
        cls,
        examples: Sequence[HighLevelTransition],
        **kwargs: object,
    ) -> SubgoalCandidateIndex:
        if not examples:
            return cls(np.zeros((0, 2)), np.zeros((0, 2)), **kwargs)  # type: ignore[arg-type]
        current = np.stack([e.h_tau[:2] for e in examples])
        future = np.stack([e.g_tau for e in examples])
        states = np.stack([e.h_tau for e in examples])
        episode_ids = np.asarray([e.episode_id for e in examples], dtype=np.int64)
        times = np.asarray([e.t for e in examples], dtype=np.int64)
        return cls(
            current,
            future,
            source_states=states,
            episode_ids=episode_ids,
            times=times,
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def from_transitions(
        cls,
        transitions: Sequence[Transition],
        horizon_k: int = DEFAULT_HORIZON_K,
        **kwargs: object,
    ) -> SubgoalCandidateIndex:
        from hwm_director.data.manager_dataset import manager_arrays_from_transitions

        states, _goals, targets, episode_ids, times = manager_arrays_from_transitions(
            transitions, horizon_k
        )
        return cls(
            states[:, :2],
            targets,
            source_states=states,
            episode_ids=episode_ids,
            times=times,
            **kwargs,  # type: ignore[arg-type]
        )

    def _nearby_indices(
        self, query_xy: np.ndarray, radius: float
    ) -> tuple[np.ndarray, np.ndarray]:
        query_xy = np.asarray(query_xy, dtype=np.float64).reshape(2)
        if self.current_xy.shape[0] == 0:
            empty = np.zeros((0,), dtype=np.int64)
            return empty, np.zeros((0,), dtype=np.float64)
        cell_span = int(np.ceil(radius / self.cell_size)) + 1
        qcell = self._cell(query_xy)
        idx_parts: list[np.ndarray] = []
        for dx in range(-cell_span, cell_span + 1):
            for dy in range(-cell_span, cell_span + 1):
                key = (qcell[0] + dx, qcell[1] + dy)
                if key in self._grid:
                    idx_parts.append(self._grid[key])
        if not idx_parts:
            empty = np.zeros((0,), dtype=np.int64)
            return empty, np.zeros((0,), dtype=np.float64)
        idx = np.concatenate(idx_parts, axis=0)
        d_xy = np.linalg.norm(self.current_xy[idx] - query_xy, axis=1)
        keep = d_xy <= float(radius)
        return idx[keep], d_xy[keep]

    def retrieve(
        self,
        current_xy: np.ndarray,
        *,
        radius: float = DEFAULT_CANDIDATE_RADIUS,
        n_candidates: int = DEFAULT_N_CANDIDATES,
        max_subgoal_distance: float = DEFAULT_MAX_SUBGOAL_DISTANCE,
    ) -> np.ndarray:
        """XY-proximity retrieval. Returns ``(M, 2)`` recorded futures."""
        records = self.retrieve_records(
            np.asarray(current_xy, dtype=np.float64).reshape(2),
            mode="xy",
            radius=radius,
            n_candidates=n_candidates,
            max_subgoal_distance=max_subgoal_distance,
        )
        if not records:
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack([r["future_xy"] for r in records], axis=0)

    def retrieve_records(
        self,
        query_state_or_xy: np.ndarray,
        *,
        mode: str = "xy",
        radius: float = DEFAULT_CANDIDATE_RADIUS,
        n_candidates: int = DEFAULT_N_CANDIDATES,
        max_subgoal_distance: float = DEFAULT_MAX_SUBGOAL_DISTANCE,
        max_source_state_distance: float | None = None,
        state_distance_weight: float = DEFAULT_STATE_DISTANCE_WEIGHT,
        xy_distance_weight: float = DEFAULT_XY_DISTANCE_WEIGHT,
        normalizer: StateNormalizer | None = None,
    ) -> list[dict]:
        """Return scored candidate records (local futures + source metadata)."""
        mode = str(mode)
        if mode not in ("xy", "state", "hybrid"):
            raise ValueError(f"unknown retrieval mode {mode!r}")
        arr = np.asarray(query_state_or_xy, dtype=np.float64).reshape(-1)
        if arr.shape[0] == 2:
            query_xy = arr
            query_state = np.zeros(STATE_DIM, dtype=np.float64)
            query_state[:2] = query_xy
        elif arr.shape[0] == STATE_DIM:
            query_state = arr
            query_xy = arr[:2]
        else:
            raise ValueError(
                f"expected xy (2,) or state ({STATE_DIM},), got {arr.shape}"
            )
        idx, d_xy = self._nearby_indices(query_xy, radius)
        if idx.size == 0:
            return []
        futures = self.future_xy[idx]
        keep = np.linalg.norm(futures - query_xy, axis=1) <= float(max_subgoal_distance)
        idx = idx[keep]
        d_xy = d_xy[keep]
        if idx.size == 0:
            return []

        used_norm = normalizer if normalizer is not None else self.normalizer
        if mode in ("state", "hybrid") and used_norm is None:
            raise ValueError("state/hybrid retrieval requires a fit StateNormalizer")
        if used_norm is not None:
            query_n = np.asarray(used_norm.normalize(query_state), dtype=np.float64)
            if self.source_states_n is not None and used_norm is self.normalizer:
                sources_n = self.source_states_n[idx]
            else:
                sources_n = np.asarray(
                    used_norm.normalize(self.source_states[idx]), dtype=np.float64
                )
            d_state = np.linalg.norm(sources_n - query_n, axis=1)
            d_xy_n = np.linalg.norm(sources_n[:, :2] - query_n[:2], axis=1)
        else:
            d_state = np.full(idx.shape, np.nan)
            d_xy_n = np.full(idx.shape, np.nan)

        if max_source_state_distance is not None and np.isfinite(d_state).any():
            keep_s = d_state <= float(max_source_state_distance)
            idx = idx[keep_s]
            d_xy = d_xy[keep_s]
            d_state = d_state[keep_s]
            d_xy_n = d_xy_n[keep_s]
            if idx.size == 0:
                return []

        if mode == "xy":
            rank = d_xy
        elif mode == "state":
            rank = d_state
        else:
            rank = (
                float(state_distance_weight) * d_state
                + float(xy_distance_weight) * d_xy_n
            )
        n = min(int(n_candidates), int(idx.size))
        if idx.size > n:
            pick = np.argpartition(rank, n - 1)[:n]
            order = np.argsort(rank[pick])
            pick = pick[order]
        else:
            pick = np.argsort(rank)
        records: list[dict] = []
        for j in pick:
            i = int(idx[j])
            records.append(
                {
                    "future_xy": self.future_xy[i].copy(),
                    "source_state": self.source_states[i].copy(),
                    "source_xy": self.current_xy[i].copy(),
                    "source_episode_id": int(self.episode_ids[i]),
                    "source_t": int(self.times[i]),
                    "source_xy_distance": float(d_xy[j]),
                    "source_state_distance": float(d_state[j]),
                    "normalized_xy_distance": float(d_xy_n[j]),
                    "hybrid_distance": float(rank[j]),
                }
            )
        return records


def estimate_source_state_distance_threshold(
    transitions: Sequence[Transition],
    normalizer: StateNormalizer,
    *,
    xy_radius: float = DEFAULT_CANDIDATE_RADIUS,
    percentile: float = DEFAULT_SOURCE_DISTANCE_PERCENTILE,
    max_pairs: int = 20000,
    seed: int = 0,
) -> dict:
    """Percentiles of normalized 29-D distance for same-episode nearby pairs.

    Nearby means same episode and torso x/y distance ``<= xy_radius``.
    Consecutive (lag-1) pairs are also reported. The recommended threshold
    is the ``percentile`` of the xy-nearby same-episode distribution so the
    cutoff is empirical, not invented.
    """
    if normalizer.mean is None or normalizer.std is None:
        raise ValueError("normalizer must be fit before estimating the threshold")
    rng = np.random.default_rng(seed)
    lag1: list[float] = []
    nearby: list[float] = []
    for traj in group_by_episode(transitions).values():
        if len(traj) < 2:
            continue
        states = np.stack([np.asarray(step.state, dtype=np.float64) for step in traj])
        states_n = normalizer.normalize(states)
        d_lag = np.linalg.norm(states_n[1:] - states_n[:-1], axis=1)
        lag1.extend(float(x) for x in d_lag)
        n = states.shape[0]
        if n < 2:
            continue
        n_sample = min(n, 64)
        picks = rng.choice(n, size=n_sample, replace=False)
        xy = states[picks, :2]
        d_xy = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
        d_st = np.linalg.norm(
            states_n[picks][:, None, :] - states_n[picks][None, :, :], axis=-1
        )
        iu = np.triu_indices(n_sample, k=1)
        mask = d_xy[iu] <= float(xy_radius)
        nearby.extend(float(x) for x in d_st[iu][mask])
        if len(nearby) >= max_pairs and len(lag1) >= max_pairs:
            break
    lag1_arr = np.asarray(lag1[:max_pairs], dtype=np.float64)
    nearby_arr = np.asarray(nearby[:max_pairs], dtype=np.float64)
    use = nearby_arr if nearby_arr.size else lag1_arr
    source = (
        "same-episode pairs with xy distance "
        f"<= {xy_radius} m"
        if nearby_arr.size
        else "consecutive lag-1 pairs (fallback; no xy-nearby pairs)"
    )
    chosen = (
        float(np.percentile(use, percentile)) if use.size else float("nan")
    )
    return {
        "percentile": float(percentile),
        "n_lag1_pairs": int(lag1_arr.size),
        "n_xy_nearby_pairs": int(nearby_arr.size),
        "lag1_p50": float(np.median(lag1_arr)) if lag1_arr.size else float("nan"),
        "lag1_p90": float(np.percentile(lag1_arr, 90)) if lag1_arr.size else float("nan"),
        "lag1_p95": float(np.percentile(lag1_arr, 95)) if lag1_arr.size else float("nan"),
        "nearby_p50": float(np.median(nearby_arr)) if nearby_arr.size else float("nan"),
        "nearby_p90": float(np.percentile(nearby_arr, 90)) if nearby_arr.size else float("nan"),
        "nearby_p95": float(np.percentile(nearby_arr, 95)) if nearby_arr.size else float("nan"),
        "chosen_threshold": chosen,
        "source": source,
    }
