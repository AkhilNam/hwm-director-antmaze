"""Train high-level value model ``Q_H`` on trajectory-aware targets.

``Q_H`` is a scorer for manager candidate selection. It is **not** ``f_H``
and does not replace Director's implicit ``(f_L, pi_L)^K``.

Training example:

    input:  h_tau, recorded g_tau = s_{t+K}[:2], final goal g*
    target: gamma ** remaining_high_level_steps   (successful episodes)
            unsuccessful_value                    (timeout / never reached g*)

Episode-level train/val split. Normalization is fit on training ``h_tau``
only. Loss is MSE.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hwm_director.data.high_level_transitions import (
    DEFAULT_SUCCESS_THRESHOLD,
    DEFAULT_UNSUCCESSFUL_VALUE,
    DEFAULT_VALUE_GAMMA,
    HighLevelValueDataset,
    build_high_level_transitions,
)
from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.transitions import Transition
from hwm_director.data.worker_dataset import DEFAULT_HORIZON_K
from hwm_director.models.high_level_value import HighLevelValueModel
from hwm_director.training.train_dynamics import split_episode_indices
from hwm_director.training.train_worker import _select, _stack_states


def _predict_values(
    model: HighLevelValueModel,
    dataset: HighLevelValueDataset,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    if len(dataset) == 0:
        return np.zeros((0,), dtype=np.float32)
    chunks: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["state"], batch["subgoal"], batch["final_goal"])
            chunks.append(pred.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_high_level_value(
    transitions: Sequence[Transition],
    model: HighLevelValueModel | None = None,
    horizon_k: int = DEFAULT_HORIZON_K,
    val_fraction: float = 0.2,
    seed: int = 0,
    batch_size: int = 4096,
    epochs: int = 20,
    lr: float = 1e-3,
    gamma: float = DEFAULT_VALUE_GAMMA,
    unsuccessful_value: float = DEFAULT_UNSUCCESSFUL_VALUE,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Fit ``Q_H`` by MSE on trajectory-aware high-level value targets."""

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

    _emit("building high-level transitions...")
    train_hl = build_high_level_transitions(
        train_raw,
        horizon_k=horizon_k,
        gamma=gamma,
        unsuccessful_value=unsuccessful_value,
        success_threshold=success_threshold,
    )
    val_hl = build_high_level_transitions(
        val_raw,
        horizon_k=horizon_k,
        gamma=gamma,
        unsuccessful_value=unsuccessful_value,
        success_threshold=success_threshold,
    )
    if not train_hl:
        raise ValueError("Training split produced no high-level examples")

    _emit(
        f"fit normalizer on {len(train_raw)} train transitions "
        f"({len(train_episode_ids)} episodes)..."
    )
    normalizer = StateNormalizer().fit(_stack_states(train_raw))
    train_dataset = HighLevelValueDataset(train_hl, normalizer=normalizer)
    val_dataset = HighLevelValueDataset(val_hl, normalizer=normalizer)

    n_train_success = int(np.sum(train_dataset.episode_succeeded))
    n_train_fail = int(len(train_dataset) - n_train_success)
    n_val_success = int(np.sum(val_dataset.episode_succeeded))
    n_val_fail = int(len(val_dataset) - n_val_success)
    _emit(
        f"train hl={len(train_dataset)} (success {n_train_success}, "
        f"unsuccessful {n_train_fail})  val hl={len(val_dataset)} "
        f"(success {n_val_success}, unsuccessful {n_val_fail})  "
        f"K={horizon_k}  gamma={gamma}  unsuccessful_value={unsuccessful_value}"
    )

    if model is None:
        model = HighLevelValueModel()

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
            pred = model(batch["state"], batch["subgoal"], batch["final_goal"])
            loss = loss_fn(pred, batch["value"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            n_batches += 1
        _emit(
            f"epoch {epoch + 1}/{epochs}  train_batch_mse={running / max(n_batches, 1):.6f}"
        )

    _emit("computing train/val value MSE...")
    train_pred = _predict_values(model, train_dataset, batch_size)
    val_pred = _predict_values(model, val_dataset, batch_size)
    train_mse = (
        float(np.mean((train_pred - train_dataset.value_targets) ** 2))
        if len(train_dataset)
        else float("nan")
    )
    val_mse = (
        float(np.mean((val_pred - val_dataset.value_targets) ** 2))
        if len(val_dataset)
        else float("nan")
    )

    return {
        "train_mse": train_mse,
        "val_mse": val_mse,
        "n_train_examples": len(train_dataset),
        "n_val_examples": len(val_dataset),
        "n_train_success_examples": n_train_success,
        "n_train_unsuccessful_examples": n_train_fail,
        "n_val_success_examples": n_val_success,
        "n_val_unsuccessful_examples": n_val_fail,
        "n_train_episodes": len(train_episode_ids),
        "n_val_episodes": len(val_episode_ids),
        "train_episode_ids": train_episode_ids,
        "val_episode_ids": val_episode_ids,
        "gamma": float(gamma),
        "unsuccessful_value": float(unsuccessful_value),
        "success_threshold": float(success_threshold),
        "horizon_k": int(horizon_k),
        "model": model,
        "normalizer": normalizer,
        "train_high_level": train_hl,
        "val_high_level": val_hl,
    }
