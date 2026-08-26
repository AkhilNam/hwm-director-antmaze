"""Train and evaluate the goal-conditioned worker ``pi_L``.

Behavior cloning: ``MSE(pi_L(s_t, g_tau), a_t)`` on episode-split data.
Normalization (states and subgoal x/y) is fit on **training episodes only**.

Actions stay in ``[-1, 1]`` and are not standardized.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import extract_state_and_goal
from hwm_director.data.trajectories import group_by_episode
from hwm_director.data.transitions import Transition
from hwm_director.data.worker_dataset import (
    DEFAULT_HORIZON_K,
    WorkerDataset,
    normalize_subgoal,
)
from hwm_director.envs.antmaze import make_antmaze
from hwm_director.models.worker import GoalConditionedWorker
from hwm_director.training.train_dynamics import split_episode_indices


def zero_action_mse(actions: np.ndarray) -> float:
    """MSE of predicting the zero torque vector.

    Parameters
    ----------
    actions:
        ``(N, 8)``.
    """
    return float(np.mean(actions**2))


def _select(
    transitions: Sequence[Transition], indices: np.ndarray
) -> list[Transition]:
    return [transitions[int(i)] for i in indices]


def _stack_states(transitions: Sequence[Transition]) -> np.ndarray:
    return np.stack([t.state for t in transitions], axis=0)


def _predict_actions(
    model: GoalConditionedWorker,
    dataset: WorkerDataset,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    if len(dataset) == 0:
        return np.zeros((0, 8), dtype=np.float32)
    chunks: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["state"], batch["subgoal"])
            chunks.append(pred.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _candidate_trials(
    transitions: Sequence[Transition], horizon_k: int
) -> list[tuple[list[Transition], int, int]]:
    """Valid ``(traj, t, k)`` with ``1 <= k <= min(K, steps remaining)``."""
    candidates: list[tuple[list[Transition], int, int]] = []
    for traj in group_by_episode(transitions).values():
        n_steps = len(traj)
        for t in range(n_steps):
            last_k = min(horizon_k, n_steps - t)
            for k in range(1, last_k + 1):
                candidates.append((traj, t, k))
    return candidates


def _set_torso_xy(env, xy: np.ndarray) -> dict:
    """Teleport the ant torso x/y after reset and return a fresh observation."""
    maze = env.unwrapped
    ant = maze.ant_env
    qpos = np.array(ant.data.qpos, copy=True)
    qvel = np.array(ant.data.qvel, copy=True)
    qpos[0] = float(xy[0])
    qpos[1] = float(xy[1])
    ant.set_state(qpos, qvel)
    return maze._get_obs(ant._get_obs())


def _worker_action(
    model: GoalConditionedWorker,
    normalizer: StateNormalizer,
    state: np.ndarray,
    subgoal_xy: np.ndarray,
) -> np.ndarray:
    state_n = normalizer.normalize(np.asarray(state))
    subgoal_n = normalize_subgoal(np.asarray(subgoal_xy), normalizer)
    state_t = torch.as_tensor(state_n, dtype=torch.float32).unsqueeze(0)
    subgoal_t = torch.as_tensor(subgoal_n, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = model(state_t, subgoal_t).squeeze(0).cpu().numpy()
    return np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)


def train_goal_conditioned_worker(
    transitions: Sequence[Transition],
    model: GoalConditionedWorker | None = None,
    horizon_k: int = DEFAULT_HORIZON_K,
    val_fraction: float = 0.2,
    seed: int = 0,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
) -> dict:
    """Fit ``pi_L`` by behavior cloning and return metrics.

    Workflow:

    1. ``split_episode_indices`` (whole episodes).
    2. Fit ``StateNormalizer`` on train states only ``(N_train, 107)``.
    3. ``WorkerDataset(..., horizon_k, normalizer)`` on train and on val.
    4. Adam + MSE on predicted vs recorded actions.
    5. Report train/val action MSE and zero-action val MSE.
    """
    if len(transitions) < 2:
        raise ValueError("Need at least 2 transitions for a train/val split")

    train_idx, val_idx = split_episode_indices(
        transitions, val_fraction=val_fraction, seed=seed
    )
    train_raw = _select(transitions, train_idx)
    val_raw = _select(transitions, val_idx)

    train_episode_ids = tuple(sorted({int(t.episode_id) for t in train_raw}))
    val_episode_ids = tuple(sorted({int(t.episode_id) for t in val_raw}))
    all_episode_ids = tuple(sorted({int(t.episode_id) for t in transitions}))

    normalizer = StateNormalizer().fit(_stack_states(train_raw))
    train_dataset = WorkerDataset(
        train_raw, horizon_k=horizon_k, normalizer=normalizer
    )
    val_dataset = WorkerDataset(
        val_raw, horizon_k=horizon_k, normalizer=normalizer
    )
    if len(train_dataset) == 0:
        raise ValueError("Training split produced no worker BC examples")

    if model is None:
        model = GoalConditionedWorker()

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
            pred = model(batch["state"], batch["subgoal"])
            loss = loss_fn(pred, batch["action"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_pred = _predict_actions(model, train_dataset, batch_size)
    val_pred = _predict_actions(model, val_dataset, batch_size)
    train_target = train_dataset.actions.numpy()
    val_target = val_dataset.actions.numpy()

    train_mse = float(np.mean((train_pred - train_target) ** 2))
    if len(val_dataset) == 0:
        val_mse = float("nan")
        zero_val = float("nan")
    else:
        val_mse = float(np.mean((val_pred - val_target) ** 2))
        zero_val = zero_action_mse(val_target)

    return {
        "train_mse": train_mse,
        "val_mse": val_mse,
        "zero_action_val_mse": zero_val,
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
    }


def evaluate_worker_on_recorded_subgoals(
    transitions: Sequence[Transition],
    model: GoalConditionedWorker,
    normalizer: StateNormalizer,
    horizon_k: int = DEFAULT_HORIZON_K,
    n_trials: int = 20,
    success_threshold: float = 0.5,
    seed: int = 0,
) -> dict:
    """Roll ``pi_L`` toward future x/y from recorded val trajectories.

    Samples ``(s_t, g)`` where ``g`` is a later x/y in the **same** episode
    (offset ``<= K``). The ant is teleported to recorded ``s_t`` x/y, then
    the worker acts in AntMaze for ``k`` steps. Distances are raw meters.
    """
    if n_trials <= 0:
        return {
            "mean_initial_distance": float("nan"),
            "mean_final_distance": float("nan"),
            "progress_fraction": float("nan"),
            "success_rate": float("nan"),
            "n_trials": 0,
        }

    candidates = _candidate_trials(transitions, horizon_k)
    if not candidates:
        raise ValueError("No in-episode (t, k) pairs to evaluate")

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(candidates), size=n_trials)
    model.eval()

    initials: list[float] = []
    finals: list[float] = []
    env = make_antmaze()
    try:
        for trial_i, cand_i in enumerate(picks):
            traj, t, k = candidates[int(cand_i)]
            start_xy = np.asarray(traj[t].state[:2], dtype=np.float64)
            subgoal = np.asarray(
                traj[t + k - 1].next_state[:2], dtype=np.float64
            )
            observation, _info = env.reset(seed=int(seed + trial_i))
            observation = _set_torso_xy(env, start_xy)
            achieved, _ = extract_state_and_goal(observation)
            initial = float(np.linalg.norm(achieved[:2] - subgoal))

            for _ in range(k):
                state, _ = extract_state_and_goal(observation)
                action = _worker_action(model, normalizer, state, subgoal)
                observation, _reward, terminated, truncated, _info = env.step(
                    action
                )
                if terminated or truncated:
                    break

            achieved, _ = extract_state_and_goal(observation)
            final = float(np.linalg.norm(achieved[:2] - subgoal))
            initials.append(initial)
            finals.append(final)
    finally:
        env.close()

    initial_arr = np.asarray(initials, dtype=np.float64)
    final_arr = np.asarray(finals, dtype=np.float64)
    return {
        "mean_initial_distance": float(np.mean(initial_arr)),
        "mean_final_distance": float(np.mean(final_arr)),
        "progress_fraction": float(np.mean(final_arr < initial_arr)),
        "success_rate": float(np.mean(final_arr < success_threshold)),
        "n_trials": int(n_trials),
    }
