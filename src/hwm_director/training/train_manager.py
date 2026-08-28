"""Train the Director manager ``pi_H`` by behavior cloning.

Loss is MSE between predicted and recorded ``s_{t+K}[:2]`` after decoding
the network's normalized relative displacement back to **raw meters**.

Baselines (raw x/y, validation):

- current-position: ``g_tau = s_t[:2]``
- direct-final-goal: ``g_tau = g*``
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hwm_director.data.manager_dataset import DirectorManagerDataset
from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.transitions import Transition
from hwm_director.data.worker_dataset import DEFAULT_HORIZON_K, denormalize_subgoal
from hwm_director.models.director_manager import DirectorManager
from hwm_director.training.train_dynamics import split_episode_indices
from hwm_director.training.train_worker import _select, _stack_states


def _predict_subgoals_raw(
    model: DirectorManager,
    dataset: DirectorManagerDataset,
    normalizer: StateNormalizer,
    batch_size: int,
    *,
    clamp: bool,
) -> np.ndarray:
    model.eval()
    if len(dataset) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    chunks: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            pred_n = model(batch["state"], batch["final_goal"], clamp=clamp)
            pred_raw = denormalize_subgoal(pred_n.cpu().numpy(), normalizer)
            chunks.append(pred_raw)
    return np.concatenate(chunks, axis=0)


def xy_mse(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((predicted - target) ** 2))


def xy_euclidean_error(predicted: np.ndarray, target: np.ndarray) -> float:
    delta = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(np.linalg.norm(delta, axis=-1)))


def train_director_manager(
    transitions: Sequence[Transition],
    model: DirectorManager | None = None,
    horizon_k: int = DEFAULT_HORIZON_K,
    val_fraction: float = 0.2,
    seed: int = 0,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Fit ``pi_H`` by BC on exactly-K future x/y from the same episode."""

    def _emit(message: str) -> None:
        if log is not None:
            log(message)

    if len(transitions) < 2:
        raise ValueError("Need at least 2 transitions for a train/val split")

    _emit("splitting episodes...")
    train_idx, val_idx = split_episode_indices(
        transitions, val_fraction=val_fraction, seed=seed
    )
    train_raw = _select(transitions, train_idx)
    val_raw = _select(transitions, val_idx)
    train_episode_ids = tuple(sorted({int(t.episode_id) for t in train_raw}))
    val_episode_ids = tuple(sorted({int(t.episode_id) for t in val_raw}))
    all_episode_ids = tuple(sorted({int(t.episode_id) for t in transitions}))

    _emit(
        f"fit normalizer on {len(train_raw)} train transitions "
        f"({len(train_episode_ids)} episodes)..."
    )
    normalizer = StateNormalizer().fit(_stack_states(train_raw))
    train_dataset = DirectorManagerDataset(
        train_raw, horizon_k=horizon_k, normalizer=normalizer
    )
    val_dataset = DirectorManagerDataset(
        val_raw, horizon_k=horizon_k, normalizer=normalizer
    )
    _emit(
        f"train examples={len(train_dataset)}  val examples={len(val_dataset)}  "
        f"K={horizon_k}  batch_size={batch_size}  epochs={epochs}"
    )
    if len(train_dataset) == 0:
        raise ValueError("Training split produced no manager BC examples")

    if model is None:
        model = DirectorManager()

    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model.train()
    for epoch in range(epochs):
        running = 0.0
        n_batches = 0
        for batch in loader:
            pred_n = model(batch["state"], batch["final_goal"], clamp=False)
            loss = loss_fn(pred_n, batch["target_subgoal"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            n_batches += 1
        _emit(
            f"epoch {epoch + 1}/{epochs}  train_batch_mse={running / max(n_batches, 1):.6f}"
        )

    _emit("computing train/val subgoal metrics...")
    train_pred = _predict_subgoals_raw(
        model, train_dataset, normalizer, batch_size, clamp=False
    )
    val_pred = _predict_subgoals_raw(
        model, val_dataset, normalizer, batch_size, clamp=False
    )
    train_target = train_dataset.raw_target_subgoals
    val_target = val_dataset.raw_target_subgoals
    train_mse = xy_mse(train_pred, train_target) if len(train_dataset) else float("nan")
    if len(val_dataset) == 0:
        val_mse = float("nan")
        val_euclid = float("nan")
        current_mse = float("nan")
        current_euclid = float("nan")
        final_mse = float("nan")
        final_euclid = float("nan")
    else:
        val_mse = xy_mse(val_pred, val_target)
        val_euclid = xy_euclidean_error(val_pred, val_target)
        current_xy = val_dataset.raw_states[:, :2]
        final_goal_xy = val_dataset.raw_final_goals
        current_mse = xy_mse(current_xy, val_target)
        current_euclid = xy_euclidean_error(current_xy, val_target)
        final_mse = xy_mse(final_goal_xy, val_target)
        final_euclid = xy_euclidean_error(final_goal_xy, val_target)

    return {
        "train_mse": train_mse,
        "val_mse": val_mse,
        "val_xy_euclidean": val_euclid,
        "current_position_val_mse": current_mse,
        "current_position_val_euclidean": current_euclid,
        "final_goal_val_mse": final_mse,
        "final_goal_val_euclidean": final_euclid,
        "n_episodes": len(all_episode_ids),
        "n_train_episodes": len(train_episode_ids),
        "n_val_episodes": len(val_episode_ids),
        "n_train_transitions": len(train_raw),
        "n_val_transitions": len(val_raw),
        "n_train_examples": len(train_dataset),
        "n_val_examples": len(val_dataset),
        "train_episode_ids": train_episode_ids,
        "val_episode_ids": val_episode_ids,
        "model": model,
        "normalizer": normalizer,
        "horizon_k": horizon_k,
    }
