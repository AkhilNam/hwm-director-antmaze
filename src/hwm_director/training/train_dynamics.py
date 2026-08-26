"""Train and evaluate one-step ``f_L``.

Normalization statistics are fit on **training states only**, then applied
to validation. Loss is MSE on predicted next-state after adding the predicted
delta, in **normalized** coordinates for both train and val.

``val_xy_mse`` is reported in raw torso x/y (meters) after denormalizing.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.torch_dataset import TransitionDataset
from hwm_director.data.transitions import Transition
from hwm_director.models.dynamics_low import LowLevelDynamicsModel


def split_indices(
    n: int, val_fraction: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Random train/validation index split with no overlap.

    Parameters
    ----------
    n:
        Number of transitions.
    val_fraction:
        Fraction assigned to validation (e.g. ``0.2``).
    seed:
        RNG seed for the permutation.

    Returns
    -------
    train_idx, val_idx
        1-D integer arrays. Disjoint, union is ``{0, ..., n-1}``.
    """
    permutation = np.random.default_rng(seed).permutation(n)
    n_val = int(round(n * val_fraction))
    if n >= 2:
        n_val = min(max(n_val, 1), n - 1)
    else:
        n_val = 0
    if n_val == 0:
        return permutation, permutation[:0]
    return permutation[:-n_val], permutation[-n_val:]


def no_change_baseline_mse(
    states: np.ndarray, next_states: np.ndarray
) -> float:
    """MSE of ``predicted_next_state = state`` (ignore actions).

    Parameters
    ----------
    states, next_states:
        Shape ``(N, 107)``.

    Returns
    -------
    float
        Mean squared error over all batch and state dimensions.
    """
    return float(np.mean((next_states - states) ** 2))


def next_position_mse(
    predicted_next_states: np.ndarray, next_states: np.ndarray
) -> float:
    """MSE on torso x/y only (state dimensions ``0:2``).

    Parameters
    ----------
    predicted_next_states, next_states:
        Shape ``(N, 107)``.
    """
    return float(
        np.mean((predicted_next_states[:, :2] - next_states[:, :2]) ** 2)
    )


def _select(
    transitions: Sequence[Transition], indices: np.ndarray
) -> list[Transition]:
    return [transitions[int(i)] for i in indices]


def _stack_states(transitions: Sequence[Transition], field: str) -> np.ndarray:
    return np.stack([getattr(t, field) for t in transitions], axis=0)


def _normalized_transitions(
    transitions: Sequence[Transition], normalizer: StateNormalizer
) -> list[Transition]:
    states = normalizer.normalize(_stack_states(transitions, "state"))
    next_states = normalizer.normalize(_stack_states(transitions, "next_state"))
    return [
        Transition(
            state=states[i],
            action=transitions[i].action,
            next_state=next_states[i],
            goal=transitions[i].goal,
        )
        for i in range(len(transitions))
    ]


def _predict_next_states(
    model: LowLevelDynamicsModel,
    dataset: TransitionDataset,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            pred = model.predict_next_state(batch["state"], batch["action"])
            chunks.append(pred.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_low_level_dynamics(
    transitions: Sequence[Transition],
    model: LowLevelDynamicsModel | None = None,
    val_fraction: float = 0.2,
    seed: int = 0,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
) -> dict:
    """Fit ``f_L`` with MSE and return a metrics dict.

    Workflow:

    1. ``split_indices`` on ``len(transitions)``.
    2. Stack train/val states; **fit** ``StateNormalizer`` on train states only.
    3. Train MLP to predict delta with MSE in normalized coordinates.
    4. Report train MSE, val MSE, no-change val MSE, val x/y MSE.

    Returns
    -------
    dict
        ``train_mse``, ``val_mse``, ``no_change_val_mse``, ``val_xy_mse``,
        plus the fitted ``model`` and ``normalizer``.
    """
    n = len(transitions)
    if n < 2:
        raise ValueError("Need at least 2 transitions for a train/val split")

    train_idx, val_idx = split_indices(n, val_fraction=val_fraction, seed=seed)
    train_raw = _select(transitions, train_idx)
    val_raw = _select(transitions, val_idx)

    normalizer = StateNormalizer().fit(_stack_states(train_raw, "state"))
    train_norm = _normalized_transitions(train_raw, normalizer)
    val_norm = _normalized_transitions(val_raw, normalizer)

    train_dataset = TransitionDataset(train_norm)
    val_dataset = TransitionDataset(val_norm)

    if model is None:
        model = LowLevelDynamicsModel()

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
    for _ in range(epochs):
        for batch in loader:
            pred = model.predict_next_state(batch["state"], batch["action"])
            loss = loss_fn(pred, batch["next_state"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_pred = _predict_next_states(model, train_dataset, batch_size)
    val_pred = _predict_next_states(model, val_dataset, batch_size)
    train_target = train_dataset.next_states.numpy()
    val_target = val_dataset.next_states.numpy()
    val_state = val_dataset.states.numpy()

    train_mse = float(np.mean((train_pred - train_target) ** 2))
    val_mse = float(np.mean((val_pred - val_target) ** 2))
    no_change = no_change_baseline_mse(val_state, val_target)
    val_xy = next_position_mse(
        normalizer.denormalize(val_pred),
        _stack_states(val_raw, "next_state"),
    )

    return {
        "train_mse": train_mse,
        "val_mse": val_mse,
        "no_change_val_mse": no_change,
        "val_xy_mse": val_xy,
        "model": model,
        "normalizer": normalizer,
    }
