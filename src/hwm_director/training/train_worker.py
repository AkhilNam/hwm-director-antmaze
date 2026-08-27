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
from hwm_director.data.state import STATE_DIM, extract_state_and_goal
from hwm_director.data.trajectories import group_by_episode
from hwm_director.data.transitions import Transition
from hwm_director.data.worker_dataset import (
    DEFAULT_HORIZON_K,
    WorkerDataset,
    normalize_subgoal,
)
from hwm_director.envs.antmaze import (
    DEFAULT_ENV_ID,
    make_antmaze,
    recover_minari_environment,
)
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
# 29-D Ant-v4 layout: [xy (2), observation (27) = qpos[2:] + qvel].
# There are no contact-force channels.
PROPRIO_SLICE = slice(2, STATE_DIM)

CLOSED_LOOP_EVAL_TODO = (
    "TODO: replace qpos/qvel restore with a valid offline/online closed-loop "
    "evaluation that does not invent MuJoCo state. Options include (1) an "
    "online AntMaze_UMaze-v4 rollout from env.reset with a goal-conditioned "
    "success metric, or (2) using a dataset that stores exact simulator "
    "state in infos. Do not approximate restore from an incomplete state."
)


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


def transitions_have_simulator_state(transitions: Sequence[Transition]) -> bool:
    """True iff every transition has both ``qpos`` and ``qvel``."""
    if not transitions:
        return False
    return all(t.qpos is not None and t.qvel is not None for t in transitions)


def restore_ant_state(env, qpos: np.ndarray, qvel: np.ndarray) -> dict:
    """Restore the recorded MuJoCo configuration and return a maze observation.

    Closed-loop eval must start from the **same physical pose** as ``s_t``.
    Writing only torso x/y onto a fresh reset leaves joints, height, and
    velocities at the default standing pose, so ``pi_L`` sees a different
    29-D state than the one it was cloned from.

    This path is valid only when ``qpos``/``qvel`` are exact (stored in the
    dataset or reconstructed from Ant-v4 via ``ant_v4_qpos_qvel_from_state``).
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
    """Compare restored 29-D state to the recorded vector.

    For Ant-v4, the full observation is proprioception (``qpos[2:]`` and
    ``qvel``). After ``set_state``, x/y and that 27-D body vector should
    match. There are no contact-force channels in this representation.
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
    }


def _assert_xy_restored(reconstructed_state: np.ndarray, recorded_state: np.ndarray) -> None:
    diag = restored_state_diagnostics(reconstructed_state, recorded_state)
    if diag["xy_abs_err"] > XY_RESTORE_TOL:
        raise RuntimeError(
            "Restored torso x/y does not match recorded state[:2] "
            f"(max abs err {diag['xy_abs_err']})"
        )


CONTROLLER_TYPES = ("worker", "zero", "random")


def controller_action(
    controller_type: str,
    *,
    model: GoalConditionedWorker | None = None,
    normalizer: StateNormalizer | None = None,
    state: np.ndarray | None = None,
    subgoal_xy: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return an 8-D action in ``[-1, 1]`` for ``worker``, ``zero``, or ``random``."""
    if controller_type == "worker":
        if model is None or normalizer is None or state is None or subgoal_xy is None:
            raise ValueError("worker controller requires model, normalizer, state, subgoal")
        return _worker_action(model, normalizer, state, subgoal_xy)
    if controller_type == "zero":
        return np.zeros(8, dtype=np.float32)
    if controller_type == "random":
        if rng is None:
            raise ValueError("random controller requires a seeded numpy Generator")
        return rng.uniform(-1.0, 1.0, size=8).astype(np.float32)
    raise ValueError(
        f"Unknown controller_type {controller_type!r}; "
        f"expected one of {CONTROLLER_TYPES}"
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
    2. Fit ``StateNormalizer`` on train states only ``(N_train, STATE_DIM)``.
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


def _reset_and_restore(env, seed: int, qpos: np.ndarray, qvel: np.ndarray) -> dict:
    """Reset the wrapper, then restore the recorded physical pose."""
    env.reset(seed=seed)
    return restore_ant_state(env, qpos, qvel)


def _rollout_from_restored(
    env,
    observation: dict,
    controller_type: str,
    k: int,
    subgoal_xy: np.ndarray,
    model: GoalConditionedWorker,
    normalizer: StateNormalizer,
    rng: np.random.Generator,
) -> float:
    """Step ``k`` times; return final distance to ``subgoal_xy``."""
    for _ in range(k):
        state, _ = extract_state_and_goal(observation)
        action = controller_action(
            controller_type,
            model=model,
            normalizer=normalizer,
            state=state,
            subgoal_xy=subgoal_xy,
            rng=rng,
        )
        observation, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
    achieved, _ = extract_state_and_goal(observation)
    return float(np.linalg.norm(achieved[:2] - subgoal_xy))


def skipped_closed_loop_eval(
    reason: str, success_threshold: float = 0.5
) -> dict:
    """Placeholder metrics when qpos/qvel restore is not scientifically valid."""
    empty_metrics = summarize_subgoal_eval(
        np.zeros(0), np.zeros(0), success_threshold
    )
    return {
        "skipped": True,
        "skip_reason": reason,
        "todo": CLOSED_LOOP_EVAL_TODO,
        "worker": dict(empty_metrics),
        "zero": dict(empty_metrics),
        "random": dict(empty_metrics),
        "n_candidates": 0,
        "n_trials": 0,
        "candidate_indices": [],
        "n_restores": 0,
        "trials": [],
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
    *,
    dataset_id: str | None = None,
    env_id: str = DEFAULT_ENV_ID,
) -> dict:
    """Compare worker / zero / random on the **same** restored (s_t, g, k) trials.

    Candidate indices are sampled once. Each controller then gets its own
    ``reset`` + ``qpos``/``qvel`` restore so rollouts cannot contaminate each
    other. Random actions come from the seeded NumPy Generator.

    If any transition is missing ``qpos``/``qvel``, this path is **skipped**
    rather than approximated. See ``CLOSED_LOOP_EVAL_TODO``.
    """
    empty_metrics = summarize_subgoal_eval(
        np.zeros(0), np.zeros(0), success_threshold
    )
    empty = {
        "skipped": False,
        "skip_reason": "",
        "todo": "",
        "worker": dict(empty_metrics),
        "zero": dict(empty_metrics),
        "random": dict(empty_metrics),
        "n_candidates": 0,
        "n_trials": 0,
        "candidate_indices": [],
        "n_restores": 0,
        "trials": [],
    }
    if n_trials <= 0:
        return empty

    if not transitions_have_simulator_state(transitions):
        reason = (
            "Closed-loop worker eval requires exact MuJoCo qpos/qvel on every "
            "transition. These transitions do not provide it, so the restore "
            "path is disabled rather than approximated from the 29-D state."
        )
        if verbose:
            print(reason)
            print(CLOSED_LOOP_EVAL_TODO)
        return skipped_closed_loop_eval(reason, success_threshold)

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
            f"in [{min_distance}, {max_distance}] m."
        )

    rng = np.random.default_rng(seed)
    picks = choose_unique_trial_indices(len(candidates), n_trials, rng)
    model.eval()

    per_ctrl: dict[str, dict[str, list[float]]] = {
        name: {"initials": [], "finals": []} for name in CONTROLLER_TYPES
    }
    trials: list[dict] = []
    n_restores = 0
    if dataset_id is not None:
        env = recover_minari_environment(dataset_id)
    else:
        env = make_antmaze(env_id)
    try:
        for trial_i, cand_i in enumerate(picks):
            cand = candidates[int(cand_i)]
            start = cand.traj[cand.t]
            if start.qpos is None or start.qvel is None:
                reason = (
                    "A sampled trial is missing qpos/qvel; refusing to invent "
                    "simulator state for closed-loop eval."
                )
                return skipped_closed_loop_eval(reason, success_threshold)
            trial_record: dict = {
                "episode_id": int(start.episode_id),
                "t": int(cand.t),
                "k": int(cand.k),
                "candidate_index": int(cand_i),
            }
            for controller_type in CONTROLLER_TYPES:
                observation = _reset_and_restore(
                    env,
                    seed=int(seed + trial_i),
                    qpos=start.qpos,
                    qvel=start.qvel,
                )
                n_restores += 1
                reconstructed, _ = extract_state_and_goal(observation)
                _assert_xy_restored(reconstructed, start.state)
                initial = float(
                    np.linalg.norm(reconstructed[:2] - cand.subgoal_xy)
                )
                final = _rollout_from_restored(
                    env,
                    observation,
                    controller_type,
                    cand.k,
                    cand.subgoal_xy,
                    model,
                    normalizer,
                    rng,
                )
                per_ctrl[controller_type]["initials"].append(initial)
                per_ctrl[controller_type]["finals"].append(final)
                trial_record[controller_type] = {
                    "initial_distance": initial,
                    "final_distance": final,
                    "distance_reduction": initial - final,
                    "started_inside_success_radius": initial < success_threshold,
                    "ended_successful": final < success_threshold,
                }
            trials.append(trial_record)
            if verbose:
                init = trial_record["worker"]["initial_distance"]
                print(f"trial ep={trial_record['episode_id']} t={trial_record['t']} k={trial_record['k']}")
                print(f"  init={init:.4f}")
                for name in CONTROLLER_TYPES:
                    rec = trial_record[name]
                    print(
                        f"  {name} final={rec['final_distance']:.4f} "
                        f"d={rec['distance_reduction']:.4f}"
                    )
    finally:
        env.close()

    result = {
        "skipped": False,
        "skip_reason": "",
        "todo": "",
        "n_candidates": len(candidates),
        "n_trials": int(picks.size),
        "candidate_indices": [int(i) for i in picks],
        "n_restores": n_restores,
        "trials": trials,
    }
    for name in CONTROLLER_TYPES:
        result[name] = summarize_subgoal_eval(
            np.asarray(per_ctrl[name]["initials"]),
            np.asarray(per_ctrl[name]["finals"]),
            success_threshold,
        )
    return result
