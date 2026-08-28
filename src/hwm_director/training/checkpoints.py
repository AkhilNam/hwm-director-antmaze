"""Save and load ``f_L`` / ``pi_L`` / ``pi_H`` / ``Q_H`` / ``f_H_phi``.

Checkpoints store:

- ``kind``: ``f_l``, ``pi_l``, ``pi_h_director``, ``high_level_value``,
  or ``f_h_explicit``
- ``state_dict``
- architecture config (dims, hidden widths, optional clamp)
- ``StateNormalizer`` ``mean`` / ``std``

``high_level_value`` is a subgoal scorer, not a high-level dynamics model.
``f_h_explicit`` is the learned coarse ``f_H_phi``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import ACHIEVED_GOAL_DIM, GOAL_DIM, STATE_DIM
from hwm_director.data.transitions import ACTION_DIM
from hwm_director.models.director_manager import (
    DEFAULT_MAX_SUBGOAL_DISTANCE,
    DirectorManager,
)
from hwm_director.models.dynamics_high import ExplicitHighLevelDynamics
from hwm_director.models.dynamics_low import LowLevelDynamicsModel
from hwm_director.models.high_level_value import HighLevelValueModel
from hwm_director.models.worker import GoalConditionedWorker

KIND_F_L = "f_l"
KIND_PI_L = "pi_l"
KIND_PI_H = "pi_h_director"
KIND_Q_H = "high_level_value"
KIND_F_H = "f_h_explicit"


def _as_path(path: str | Path) -> Path:
    return Path(path)


def save_checkpoint(
    path: str | Path,
    *,
    kind: str,
    model: torch.nn.Module,
    normalizer: StateNormalizer,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a torch checkpoint (creates parent directories)."""
    if normalizer.mean is None or normalizer.std is None:
        raise ValueError("normalizer must be fit before save_checkpoint()")
    dest = _as_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "kind": kind,
        "state_dict": model.state_dict(),
        "normalizer_mean": np.asarray(normalizer.mean),
        "normalizer_std": np.asarray(normalizer.std),
        "normalizer_eps": float(normalizer.eps),
        "config": {
            "state_dim": int(getattr(model, "state_dim", STATE_DIM)),
            "hidden_dims": tuple(getattr(model, "hidden_dims", (256, 256))),
        },
    }
    if extra:
        payload["config"].update(extra)
    torch.save(payload, dest)


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    return torch.load(_as_path(path), map_location="cpu", weights_only=False)


def normalizer_from_payload(payload: dict[str, Any]) -> StateNormalizer:
    normalizer = StateNormalizer(eps=float(payload.get("normalizer_eps", 1e-8)))
    normalizer.mean = np.asarray(payload["normalizer_mean"], dtype=np.float64)
    normalizer.std = np.asarray(payload["normalizer_std"], dtype=np.float64)
    return normalizer


def _hidden_dims(config: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(x) for x in config.get("hidden_dims", (256, 256)))


def save_dynamics_checkpoint(
    path: str | Path,
    model: LowLevelDynamicsModel,
    normalizer: StateNormalizer,
) -> None:
    save_checkpoint(
        path,
        kind=KIND_F_L,
        model=model,
        normalizer=normalizer,
        extra={
            "action_dim": int(model.action_dim),
        },
    )


def load_dynamics_checkpoint(
    path: str | Path,
) -> tuple[LowLevelDynamicsModel, StateNormalizer]:
    payload = load_checkpoint_payload(path)
    if payload.get("kind") not in (None, KIND_F_L):
        raise ValueError(f"expected kind={KIND_F_L!r}, got {payload.get('kind')!r}")
    config = payload.get("config", {})
    model = LowLevelDynamicsModel(
        state_dim=int(config.get("state_dim", STATE_DIM)),
        action_dim=int(config.get("action_dim", ACTION_DIM)),
        hidden_dims=_hidden_dims(config),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, normalizer_from_payload(payload)


def save_worker_checkpoint(
    path: str | Path,
    model: GoalConditionedWorker,
    normalizer: StateNormalizer,
) -> None:
    save_checkpoint(
        path,
        kind=KIND_PI_L,
        model=model,
        normalizer=normalizer,
        extra={
            "subgoal_dim": int(model.subgoal_dim),
            "action_dim": int(model.action_dim),
        },
    )


def load_worker_checkpoint(
    path: str | Path,
) -> tuple[GoalConditionedWorker, StateNormalizer]:
    payload = load_checkpoint_payload(path)
    if payload.get("kind") not in (None, KIND_PI_L):
        raise ValueError(f"expected kind={KIND_PI_L!r}, got {payload.get('kind')!r}")
    config = payload.get("config", {})
    model = GoalConditionedWorker(
        state_dim=int(config.get("state_dim", STATE_DIM)),
        subgoal_dim=int(config.get("subgoal_dim", ACHIEVED_GOAL_DIM)),
        action_dim=int(config.get("action_dim", ACTION_DIM)),
        hidden_dims=_hidden_dims(config),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, normalizer_from_payload(payload)


def save_manager_checkpoint(
    path: str | Path,
    model: DirectorManager,
    normalizer: StateNormalizer,
) -> None:
    save_checkpoint(
        path,
        kind=KIND_PI_H,
        model=model,
        normalizer=normalizer,
        extra={
            "goal_dim": int(model.goal_dim),
            "subgoal_dim": int(model.subgoal_dim),
            "max_subgoal_distance": float(model.max_subgoal_distance),
        },
    )


def load_manager_checkpoint(
    path: str | Path,
) -> tuple[DirectorManager, StateNormalizer]:
    payload = load_checkpoint_payload(path)
    if payload.get("kind") not in (None, KIND_PI_H):
        raise ValueError(f"expected kind={KIND_PI_H!r}, got {payload.get('kind')!r}")
    config = payload.get("config", {})
    model = DirectorManager(
        state_dim=int(config.get("state_dim", STATE_DIM)),
        goal_dim=int(config.get("goal_dim", GOAL_DIM)),
        subgoal_dim=int(config.get("subgoal_dim", ACHIEVED_GOAL_DIM)),
        hidden_dims=_hidden_dims(config),
        max_subgoal_distance=float(
            config.get("max_subgoal_distance", DEFAULT_MAX_SUBGOAL_DISTANCE)
        ),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, normalizer_from_payload(payload)


def save_high_level_value_checkpoint(
    path: str | Path,
    model: HighLevelValueModel,
    normalizer: StateNormalizer,
    *,
    gamma: float,
    horizon_k: int,
    unsuccessful_value: float = 0.0,
    success_threshold: float = 0.5,
    candidate_state_radius: float = 0.75,
    n_candidates: int = 32,
    max_subgoal_distance: float = DEFAULT_MAX_SUBGOAL_DISTANCE,
) -> None:
    save_checkpoint(
        path,
        kind=KIND_Q_H,
        model=model,
        normalizer=normalizer,
        extra={
            "goal_dim": int(model.goal_dim),
            "subgoal_dim": int(model.subgoal_dim),
            "gamma": float(gamma),
            "horizon_k": int(horizon_k),
            "unsuccessful_value": float(unsuccessful_value),
            "success_threshold": float(success_threshold),
            "candidate_state_radius": float(candidate_state_radius),
            "n_candidates": int(n_candidates),
            "max_subgoal_distance": float(max_subgoal_distance),
            "is_high_level_dynamics": False,
        },
    )


def load_high_level_value_checkpoint(
    path: str | Path,
) -> tuple[HighLevelValueModel, StateNormalizer, dict[str, Any]]:
    payload = load_checkpoint_payload(path)
    if payload.get("kind") not in (None, KIND_Q_H):
        raise ValueError(f"expected kind={KIND_Q_H!r}, got {payload.get('kind')!r}")
    config = payload.get("config", {})
    model = HighLevelValueModel(
        state_dim=int(config.get("state_dim", STATE_DIM)),
        subgoal_dim=int(config.get("subgoal_dim", ACHIEVED_GOAL_DIM)),
        goal_dim=int(config.get("goal_dim", GOAL_DIM)),
        hidden_dims=_hidden_dims(config),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, normalizer_from_payload(payload), dict(config)


def save_high_level_dynamics_checkpoint(
    path: str | Path,
    model: ExplicitHighLevelDynamics,
    normalizer: StateNormalizer,
    *,
    horizon_k: int,
    dataset_id: str = "",
    seed: int = 0,
    val_fraction: float = 0.2,
) -> None:
    save_checkpoint(
        path,
        kind=KIND_F_H,
        model=model,
        normalizer=normalizer,
        extra={
            "subgoal_dim": int(model.subgoal_dim),
            "horizon_k": int(horizon_k),
            "dataset_id": str(dataset_id),
            "seed": int(seed),
            "val_fraction": float(val_fraction),
            "is_high_level_dynamics": True,
            "training_target": "recorded s_{t+K}",
        },
    )


def load_high_level_dynamics_checkpoint(
    path: str | Path,
) -> tuple[ExplicitHighLevelDynamics, StateNormalizer, dict[str, Any]]:
    payload = load_checkpoint_payload(path)
    if payload.get("kind") not in (None, KIND_F_H):
        raise ValueError(f"expected kind={KIND_F_H!r}, got {payload.get('kind')!r}")
    config = payload.get("config", {})
    model = ExplicitHighLevelDynamics(
        state_dim=int(config.get("state_dim", STATE_DIM)),
        subgoal_dim=int(config.get("subgoal_dim", ACHIEVED_GOAL_DIM)),
        hidden_dims=_hidden_dims(config),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, normalizer_from_payload(payload), dict(config)
