"""Train and evaluate the goal-conditioned worker ``pi_L``.

Behavior cloning: ``MSE(pi_L(s_t, g_tau), a_t)`` on episode-split data.
Normalization (states and subgoal x/y) is fit on **training episodes only**.

Actions stay in ``[-1, 1]`` and are not standardized.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

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


DEFAULT_MIN_SUBGOAL_DISTANCE = 0.5
DEFAULT_MAX_SUBGOAL_DISTANCE = 2.0
XY_RESTORE_TOL = 1e-4
# 107-D layout: [xy (2), Ant-v5 proprio (27) = qpos[2:]+qvel, cfrc_ext (78)].
PROPRIO_SLICE = slice(2, 29)
CONTACT_SLICE = slice(29, 107)


class SubgoalCandidate(NamedTuple):
    """One in-episode future x/y target for closed-loop worker eval."""

    traj: list[Transition]
    t: int
    k: int
    initial_distance: float
    start_xy: np.ndarray
    subgoal_xy: np.ndarray


def _future_xy(traj: Sequence[Transition], t: int, k: int) -> np.ndarray:
    return np.asarray(traj[t + k - 1].next_state[:2], dtype=np.float64)


def subgoal_candidates(
    transitions: Sequence[Transition],
    horizon_k: int,
    min_distance: float = DEFAULT_MIN_SUBGOAL_DISTANCE,
    max_distance: float = DEFAULT_MAX_SUBGOAL_DISTANCE,
) -> list[SubgoalCandidate]:
    """In-episode ``(t, k)`` pairs with initial x/y distance in ``[min, max]``.

    Requires ``1 <= k <= K`` and never crosses episode boundaries.
    """
    if min_distance > max_distance:
        raise ValueError(
            f"min_distance ({min_distance}) > max_distance ({max_distance})"
        )
    candidates: list[SubgoalCandidate] = []
    for traj in group_by_episode(transitions).values():
        traj = list(traj)
        n_steps = len(traj)
        for t in range(n_steps):
            last_k = min(horizon_k, n_steps - t)
            start_xy = np.asarray(traj[t].state[:2], dtype=np.float64)
            for k in range(1, last_k + 1):
                subgoal_xy = _future_xy(traj, t, k)
                dist = float(np.linalg.norm(start_xy - subgoal_xy))
                if min_distance <= dist <= max_distance:
                    candidates.append(
                        SubgoalCandidate(
                            traj=traj,
                            t=t,
                            k=k,
                            initial_distance=dist,
                            start_xy=start_xy,
                            subgoal_xy=subgoal_xy,
                        )
                    )
    return candidates


def choose_unique_trial_indices(
    n_candidates: int, n_trials: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample trial indices without replacement (at most ``n_candidates``)."""
    if n_candidates <= 0:
        return np.zeros(0, dtype=np.int64)
    n = min(int(n_trials), int(n_candidates))
    return rng.choice(n_candidates, size=n, replace=False)


def summarize_subgoal_eval(
    initials: np.ndarray,
    finals: np.ndarray,
    success_threshold: float,
) -> dict:
    """Aggregate closed-loop distances (meters)."""
    initials = np.asarray(initials, dtype=np.float64)
    finals = np.asarray(finals, dtype=np.float64)
    if initials.size == 0:
        return {
            "n_trials": 0,
            "mean_initial_distance": float("nan"),
            "mean_final_distance": float("nan"),
            "progress_fraction": float("nan"),
            "success_rate": float("nan"),
            "mean_distance_reduction": float("nan"),
            "median_distance_reduction": float("nan"),
            "fraction_positive_reduction": float("nan"),
            "fraction_relative_progress_10": float("nan"),
            "fraction_already_successful_at_start": float("nan"),
        }
    reduction = initials - finals
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(initials > 0.0, reduction / initials, 0.0)
    return {
        "n_trials": int(initials.size),
        "mean_initial_distance": float(np.mean(initials)),
        "mean_final_distance": float(np.mean(finals)),
        "progress_fraction": float(np.mean(finals < initials)),
        "success_rate": float(np.mean(finals < success_threshold)),
        "mean_distance_reduction": float(np.mean(reduction)),
        "median_distance_reduction": float(np.median(reduction)),
        "fraction_positive_reduction": float(np.mean(reduction > 0.0)),
        "fraction_relative_progress_10": float(np.mean(relative >= 0.1)),
        "fraction_already_successful_at_start": float(
            np.mean(initials < success_threshold)
        ),
    }


def restore_ant_state(env, qpos: np.ndarray, qvel: np.ndarray) -> dict:
    """Restore the recorded MuJoCo configuration and return a maze observation.

    Closed-loop eval must start from the **same physical pose** as ``s_t``.
    Writing only torso x/y onto a fresh reset leaves joints, height, and
    velocities at the default standing pose, so ``pi_L`` sees a different
    107-D state than the one it was cloned from.
    """
    maze = env.unwrapped
    ant = maze.ant_env
    ant.set_state(
        np.asarray(qpos, dtype=np.float64),
        np.asarray(qvel, dtype=np.float64),
    )
    return maze._get_obs(ant._get_obs())


def restored_state_diagnostics(
    reconstructed_state: np.ndarray, recorded_state: np.ndarray
) -> dict:
    """Compare restored 107-D state to the recorded vector.

    x/y and proprioception (z, orientation, joints, qvel) should match after
    ``set_state``. Contact-force channels ``state[29:107]`` (Ant-v5 ``cfrc_ext``)
    can differ because they depend on the last collision resolution, not only
    on qpos/qvel. Those 78 dims are documented, not silently treated as equal.
    """
    reconstructed_state = np.asarray(reconstructed_state, dtype=np.float64)
    recorded_state = np.asarray(recorded_state, dtype=np.float64)
    return {
        "xy_abs_err": float(
            np.max(np.abs(reconstructed_state[:2] - recorded_state[:2]))
        ),
        "proprio_abs_err": float(
            np.max(
                np.abs(
                    reconstructed_state[PROPRIO_SLICE] - recorded_state[PROPRIO_SLICE]
                )
            )
        ),
        "contact_abs_err": float(
            np.max(
                np.abs(
                    reconstructed_state[CONTACT_SLICE] - recorded_state[CONTACT_SLICE]
                )
            )
        ),
    }


def _assert_xy_restored(reconstructed_state: np.ndarray, recorded_state: np.ndarray) -> None:
    diag = restored_state_diagnostics(reconstructed_state, recorded_state)
    if diag["xy_abs_err"] > XY_RESTORE_TOL:
        raise RuntimeError(
            "Restored torso x/y does not match recorded state[:2] "
            f"(max abs err {diag['xy_abs_err']})"
        )


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
    min_distance: float = DEFAULT_MIN_SUBGOAL_DISTANCE,
    max_distance: float = DEFAULT_MAX_SUBGOAL_DISTANCE,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Roll ``pi_L`` toward a nontrivial future x/y from recorded trajectories.

    Restores recorded ``qpos``/``qvel`` so the worker starts in the true
    physical pose, not a reset ant with only x/y overwritten. Subgoals are
    filtered to ``[min_distance, max_distance]`` meters. Trials are unique
    ``(episode, t, k)`` pairs (no replacement).
    """
    empty = summarize_subgoal_eval(
        np.zeros(0), np.zeros(0), success_threshold
    )
    empty["n_candidates"] = 0
    empty["trials"] = []
    if n_trials <= 0:
        return empty

    candidates = subgoal_candidates(
        transitions,
        horizon_k=horizon_k,
        min_distance=min_distance,
        max_distance=max_distance,
    )
    if not candidates:
        raise ValueError(
            "No eligible subgoal candidates in these trajectories. "
            f"Need in-episode offsets 1..{horizon_k} with initial x/y distance "
            f"in [{min_distance}, {max_distance}] m. Random short rollouts "
            "often stay closer than min_distance; collect longer/more diverse "
            "trajectories or relax the distance window."
        )

    rng = np.random.default_rng(seed)
    picks = choose_unique_trial_indices(len(candidates), n_trials, rng)
    model.eval()

    initials: list[float] = []
    finals: list[float] = []
    trials: list[dict] = []
    env = make_antmaze()
    try:
        for trial_i, cand_i in enumerate(picks):
            cand = candidates[int(cand_i)]
            start = cand.traj[cand.t]
            if start.qpos is None or start.qvel is None:
                raise ValueError(
                    "Transition is missing qpos/qvel; recollect transitions "
                    "with the updated collector before closed-loop eval."
                )
            observation, _info = env.reset(seed=int(seed + trial_i))
            observation = restore_ant_state(env, start.qpos, start.qvel)
            reconstructed, _ = extract_state_and_goal(observation)
            _assert_xy_restored(reconstructed, start.state)

            initial = float(np.linalg.norm(reconstructed[:2] - cand.subgoal_xy))
            started_inside = initial < success_threshold

            for _ in range(cand.k):
                state, _ = extract_state_and_goal(observation)
                action = _worker_action(
                    model, normalizer, state, cand.subgoal_xy
                )
                observation, _reward, terminated, truncated, _info = env.step(
                    action
                )
                if terminated or truncated:
                    break

            achieved, _ = extract_state_and_goal(observation)
            final = float(np.linalg.norm(achieved[:2] - cand.subgoal_xy))
            reduction = initial - final
            ended_success = final < success_threshold
            record = {
                "episode_id": int(start.episode_id),
                "t": int(cand.t),
                "k": int(cand.k),
                "initial_distance": initial,
                "final_distance": final,
                "distance_reduction": reduction,
                "started_inside_success_radius": started_inside,
                "ended_successful": ended_success,
            }
            if verbose:
                print(
                    "  trial "
                    f"ep={record['episode_id']} t={record['t']} k={record['k']} "
                    f"init={initial:.4f} final={final:.4f} "
                    f"d={reduction:.4f} start_ok={started_inside} "
                    f"end_ok={ended_success}"
                )
            initials.append(initial)
            finals.append(final)
            trials.append(record)
    finally:
        env.close()

    summary = summarize_subgoal_eval(
        np.asarray(initials), np.asarray(finals), success_threshold
    )
    summary["n_candidates"] = len(candidates)
    summary["trials"] = trials
    return summary
