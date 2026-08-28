"""Train explicit high-level dynamics ``f_H_phi`` on recorded K-step tuples.

Each example is an exactly-K offline transition (same episode):

    h_tau  = s_t
    g_tau  = s_{t+K}[:2]
    h_next = s_{t+K}

This is supervised on **recorded** behavior. It is not a closed-loop
rollout of the current learned ``pi_L``.

Normalization is fit on training-episode primitive states only:

    h_tau_n  = normalize(h_tau)
    g_tau_n  = normalize_subgoal(g_tau)
    h_next_n = normalize(h_next)
    delta_n  = h_next_n - h_tau_n

Loss is MSE between ``h_tau_n + predicted_delta`` and ``h_next_n``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hwm_director.data.high_level_transitions import (
    HighLevelDynamicsDataset,
    build_high_level_transitions,
)
from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.transitions import Transition
from hwm_director.data.worker_dataset import DEFAULT_HORIZON_K
from hwm_director.models.dynamics_high import ExplicitHighLevelDynamics
from hwm_director.training.train_dynamics import (
    next_position_mse,
    no_change_baseline_mse,
    split_episode_indices,
)
from hwm_director.training.train_worker import _select, _stack_states


def _predict_next_normalized(
    model: ExplicitHighLevelDynamics,
    dataset: HighLevelDynamicsDataset,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    if len(dataset) == 0:
        return np.zeros((0, model.state_dim), dtype=np.float32)
    chunks: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            pred = model.predict_next_state(batch["state"], batch["subgoal"])
            chunks.append(pred.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_high_level_dynamics(
    transitions: Sequence[Transition],
    model: ExplicitHighLevelDynamics | None = None,
    horizon_k: int = DEFAULT_HORIZON_K,
    val_fraction: float = 0.2,
    seed: int = 0,
    batch_size: int = 4096,
    epochs: int = 20,
    lr: float = 1e-3,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Fit ``f_H_phi`` by MSE on recorded K-step next states."""

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
    if set(train_episode_ids) & set(val_episode_ids):
        raise AssertionError("train/val episode ids overlap")

    _emit("building exactly-K high-level transitions...")
    train_hl = build_high_level_transitions(train_raw, horizon_k=horizon_k)
    val_hl = build_high_level_transitions(val_raw, horizon_k=horizon_k)
    if not train_hl:
        raise ValueError("Training split produced no high-level examples")

    _emit(
        f"fit normalizer on {len(train_raw)} train transitions "
        f"({len(train_episode_ids)} episodes)..."
    )
    normalizer = StateNormalizer().fit(_stack_states(train_raw))
    train_dataset = HighLevelDynamicsDataset(train_hl, normalizer=normalizer)
    val_dataset = HighLevelDynamicsDataset(val_hl, normalizer=normalizer)

    if model is None:
        model = ExplicitHighLevelDynamics()

    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    _emit(
        f"f_H: training {epochs} epochs, {len(train_dataset)} examples, "
        f"batch_size={batch_size}"
    )
    model.train()
    for epoch in range(epochs):
        running = 0.0
        n_batches = 0
        for batch in loader:
            pred = model.predict_next_state(batch["state"], batch["subgoal"])
            loss = loss_fn(pred, batch["next_state"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
        _emit(f"  epoch {epoch + 1}/{epochs}  train_mse={running / max(n_batches, 1):.6f}")

    train_pred_n = _predict_next_normalized(model, train_dataset, batch_size)
    val_pred_n = _predict_next_normalized(model, val_dataset, batch_size)
    train_mse = (
        float(np.mean((train_pred_n - train_dataset.h_next.numpy()) ** 2))
        if len(train_dataset)
        else float("nan")
    )
    val_mse = (
        float(np.mean((val_pred_n - val_dataset.h_next.numpy()) ** 2))
        if len(val_dataset)
        else float("nan")
    )
    if len(val_dataset):
        val_pred_raw = normalizer.denormalize(val_pred_n)
        val_true_raw = val_dataset.raw_h_next
        val_xy_mse = next_position_mse(val_pred_raw, val_true_raw)
        val_xy_euclidean = float(
            np.mean(np.linalg.norm(val_pred_raw[:, :2] - val_true_raw[:, :2], axis=1))
        )
        no_change_xy = float(
            np.mean(
                np.linalg.norm(
                    val_dataset.raw_h_tau[:, :2] - val_true_raw[:, :2], axis=1
                )
            )
        )
        no_change_mse = no_change_baseline_mse(val_dataset.raw_h_tau, val_true_raw)
    else:
        val_xy_mse = float("nan")
        val_xy_euclidean = float("nan")
        no_change_xy = float("nan")
        no_change_mse = float("nan")

    model.eval()
    return {
        "model": model,
        "normalizer": normalizer,
        "horizon_k": int(horizon_k),
        "n_episodes": len(train_episode_ids) + len(val_episode_ids),
        "n_train_episodes": len(train_episode_ids),
        "n_val_episodes": len(val_episode_ids),
        "n_train_examples": len(train_dataset),
        "n_val_examples": len(val_dataset),
        "train_episode_ids": train_episode_ids,
        "val_episode_ids": val_episode_ids,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "val_xy_mse": val_xy_mse,
        "val_xy_euclidean": val_xy_euclidean,
        "no_change_val_xy_euclidean": no_change_xy,
        "no_change_val_mse": no_change_mse,
        "train_high_level": train_hl,
        "val_high_level": val_hl,
        "target": "recorded s_{t+K} (offline behavior, not current pi_L)",
    }
