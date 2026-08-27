# hwm-director-antmaze

Compare **Director** and an explicit **Hierarchical World Model (HWM)** on AntMaze.

This repository currently implements only the shared low-level stack
\(E\), \(\pi_L\), and \(f_L\). There is **no RSSM**, **no JEPA**, **no Director**,
and **no HWM** yet. Those systems will later share this exact dataset and
representation.

## Setup

Gymnasium-Robotics documents Python 3.10–3.13. Use a 3.12 or 3.13 virtualenv if the system Python is newer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Current primary experiment

- Offline dataset: Minari [`D4RL/antmaze/umaze-v1`](https://minari.farama.org/datasets/D4RL/antmaze/umaze-v1/)
- Matching environment: `AntMaze_UMaze-v4` (recovered from the Minari dataset when possible)
- Package: [`gymnasium-robotics==1.4.2`](https://pypi.org/project/gymnasium-robotics/)
- API: Gymnasium (`observation, reward, terminated, truncated, info`)
- Physics: official `mujoco` Python bindings (not `mujoco-py`)

Raw observations are a Dict:

| Key | Shape | Meaning |
| --- | --- | --- |
| `observation` | `(27,)` | Ant-v4 proprioception (no torso x/y, no contact forces) |
| `achieved_goal` | `(2,)` | current torso x/y |
| `desired_goal` | `(2,)` | task target x/y |

## State representation

The low-level world model \(f_L\) consumes one-step tuples \((s_t, a_t, s_{t+1})\), where

- \(s_t = [\texttt{achieved\_goal},\; \texttt{observation}]\) with shape \((29,)\)
- \(g^\star = \texttt{desired\_goal}\) with shape \((2,)\)
- \(a_t\) has shape `(8,)`
- \(E(s_t) = s_t\) via `IdentityEncoder` (NumPy copy; not learned)

This is a fixed, simple representation. Dimensions are defined once in
`src/hwm_director/data/state.py` (`STATE_DIM = ACHIEVED_GOAL_DIM + OBSERVATION_DIM`).

Ant-v4 stores enough proprioception to reconstruct MuJoCo `qpos`/`qvel`
exactly from the 29-D state (`qpos = state[0:15]`, `qvel = state[15:29]`;
see `ant_v4_qpos_qvel_from_state`). **In addition**, Minari
`D4RL/antmaze/umaze-v1` episode `infos` already store exact `qpos` (15,)
and `qvel` (14,) at every observation. The loader prefers those recorded
arrays when present, and falls back to the Ant-v4 mapping otherwise.
Closed-loop worker eval therefore restores the recorded MuJoCo pose
rather than approximating it.

## Why this replaced v5 / random rollouts

Random AntMaze_UMaze-v5 rollouts were useful for debugging \(f_L\) (one-step
delta prediction still has a signal under random torques). They were **not**
suitable supervision for goal-conditioned \(\pi_L\): cloning random actions
does not teach a worker to move toward a future x/y.

Minari `D4RL/antmaze/umaze-v1` provides a **fixed offline** corpus of
goal-directed UMaze trajectories (waypoint + SAC), with a matching
`AntMaze_UMaze-v4` environment and a 27-D Ant observation (no 78-D contact
forces). Director and HWM will later share this same dataset and 29-D
state.

`scripts/collect_random_transitions.py` remains only as a debugging /
smoke-test utility.

```bash
python scripts/inspect_minari_antmaze.py
python scripts/inspect_antmaze.py
pytest
```

## Notation

| Symbol | Meaning | Status |
| --- | --- | --- |
| \(E\) | encoder | identity only |
| \(\pi_L\) | low-level worker | implemented (BC toward future x/y) |
| \(f_L\) | one-step world model | implemented (delta MLP) |
| \(\pi_H\) | high-level controller | not implemented |
| \(f_H\) | high-level model | not implemented |

## One-step \(f_L\)

**Goal:** learn primitive one-step dynamics

\[
f_L(s_t, a_t) \rightarrow \hat s_{t+1}
\]

with \(E(s_t) = s_t\) (identity; 29-D baseline state).

This is **not** Director and **not** HWM. Both of those systems will reuse the same \(f_L\). RSSM and JEPA are deferred on purpose so dynamics quality is not mixed with representation learning.

The network predicts a **state delta** \(\hat s_{t+1} - s_t\) (implementation choice). Wu’s framework only requires one-step prediction, not residual form.

```bash
python scripts/train_low_level_dynamics.py --max-episodes 8 --epochs 5
python scripts/train_low_level_dynamics.py
pytest
```

Train/validation splits are **trajectory/episode-level**: adjacent transitions from the same rollout are never divided across splits. Normalization statistics are fit only on training episodes.

## Shared worker \(\pi_L\)

**Goal:** implement the remaining shared low-level piece (with \(E\) and \(f_L\)):

\[
\pi_L(s_t, g_\tau) \rightarrow a_t
\]

First baseline: \(g_\tau \in \mathbb{R}^2\) is a **target future x/y** from the same episode, at most \(K\) steps ahead (default \(K=10\)). Training is behavior cloning of the recorded action \(a_t\). Input dimension is \(29 + 2 = 31\).

This worker is **shared by Director and HWM**. There is still no high-level policy \(\pi_H\), no explicit \(f_H\), and no RSSM/JEPA.

```bash
python scripts/train_worker.py --max-episodes 8 --epochs 5 --n-eval-trials 0
python scripts/train_worker.py --horizon-k 10
pytest
```

Closed-loop eval restores the **recorded MuJoCo pose** (`qpos`/`qvel` at `s_t`) when that state is available (Minari v4 reconstruction or live collection). Subgoals are kept only if initial x/y distance is in `[--min-subgoal-distance, --max-subgoal-distance]` (defaults 0.5–2.0 m). Each sampled trial is rolled out **three times from the same restore**: learned `π_L`, zero torques, and uniform random actions.

If exact simulator state is missing, that eval path is **disabled** (not approximated). Behavior-cloning validation metrics are still reported.

## Full experiment commands

After `pip install -e ".[dev]"`:

```bash
python scripts/inspect_minari_antmaze.py
python scripts/train_low_level_dynamics.py
python scripts/train_worker.py --horizon-k 10
pytest
```

Debug subsets:

```bash
python scripts/train_low_level_dynamics.py --max-episodes 8 --epochs 5
python scripts/train_worker.py --max-episodes 8 --epochs 5 --n-eval-trials 0
```
