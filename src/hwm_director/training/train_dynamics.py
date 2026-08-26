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


def split_episode_indices(
    transitions: Sequence[Transition],
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Train/validation split at the episode (trajectory) level.

    Unique episode IDs are shuffled with ``seed``. Whole episodes go to train
    or validation, so adjacent steps ``(s_t, a_t, s_{t+1})`` and
    ``(s_{t+1}, a_{t+1}, s_{t+2})`` never cross the split.

    ``val_fraction`` is the fraction of **episodes**, not transitions.

    Returns
    -------
    train_idx, val_idx
        Transition indices. Disjoint, union is ``{0, ..., n-1}``.

    Raises
    ------
    ValueError
        If fewer than two unique episodes are present.
    """
    n = len(transitions)
    if n == 0:
        raise ValueError("Need at least one transition to split")

    episode_ids = np.array(sorted({int(t.episode_id) for t in transitions}))
    n_episodes = int(episode_ids.size)
    if n_episodes < 2:
        raise ValueError(
            "Need at least 2 unique episodes for a train/val split, "
            f"got {n_episodes} from {n} transitions. "
            "AntMaze_UMaze-v5 episodes often run ~700 steps, so "
            "--n-transitions 256 is usually a single unfinished episode. "
            "Use at least ~1500–5000 transitions (or more) so collection "
            "covers two resets."
        )

    n_val = int(round(n_episodes * val_fraction))
    n_val = min(max(n_val, 1), n_episodes - 1)

    shuffled = np.random.default_rng(seed).permutation(episode_ids)
    val_episodes = set(int(x) for x in shuffled[-n_val:])
    train_episodes = set(int(x) for x in shuffled[:-n_val])

    train_idx = np.array(
        [i for i, t in enumerate(transitions) if int(t.episode_id) in train_episodes],
        dtype=np.int64,
    )
    val_idx = np.array(
        [i for i, t in enumerate(transitions) if int(t.episode_id) in val_episodes],
        dtype=np.int64,
    )
    return train_idx, val_idx


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
            episode_id=transitions[i].episode_id,
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

    1. ``split_episode_indices`` so whole trajectories stay in one split.
    2. Stack train/val states; **fit** ``StateNormalizer`` on train states only.
    3. Train MLP to predict delta with MSE in normalized coordinates.
    4. Report train MSE, val MSE, no-change val MSE, val x/y MSE, and
       episode/transition counts.

    Returns
    -------
    dict
        ``train_mse``, ``val_mse``, ``no_change_val_mse``, ``val_xy_mse``,
        split diagnostics, plus the fitted ``model`` and ``normalizer``.
    """
    n = len(transitions)
    if n < 2:
        raise ValueError("Need at least 2 transitions for a train/val split")

    train_idx, val_idx = split_episode_indices(
        transitions, val_fraction=val_fraction, seed=seed
    )
    train_raw = _select(transitions, train_idx)
    val_raw = _select(transitions, val_idx)

    train_episode_ids = tuple(sorted({int(t.episode_id) for t in train_raw}))
    val_episode_ids = tuple(sorted({int(t.episode_id) for t in val_raw}))
    all_episode_ids = tuple(sorted({int(t.episode_id) for t in transitions}))

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
        "n_episodes": len(all_episode_ids),
        "n_train_episodes": len(train_episode_ids),
        "n_val_episodes": len(val_episode_ids),
        "n_train_transitions": len(train_raw),
        "n_val_transitions": len(val_raw),
        "train_episode_ids": train_episode_ids,
        "val_episode_ids": val_episode_ids,
        "model": model,
        "normalizer": normalizer,
    }
