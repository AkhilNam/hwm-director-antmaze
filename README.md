# hwm-director-antmaze

Compare **Director** and an explicit **Hierarchical World Model (HWM)** on AntMaze.

## Setup

Gymnasium-Robotics documents Python 3.10–3.13. Use a 3.12 or 3.13 virtualenv if the system Python is newer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Environment

- Package: [`gymnasium-robotics==1.4.2`](https://pypi.org/project/gymnasium-robotics/)
- Env ID: `AntMaze_UMaze-v5`
- API: Gymnasium (`observation, reward, terminated, truncated, info`)
- Physics: official `mujoco` Python bindings (not `mujoco-py`)

Raw observations are a Dict:

| Key | Shape | Meaning |
| --- | --- | --- |
| `observation` | `(105,)` | Ant-v5 body state (no torso x/y) |
| `achieved_goal` | `(2,)` | current torso x/y |
| `desired_goal` | `(2,)` | task target x/y |

## State representation

The low-level world model \(f_L\) consumes one-step tuples \((s_t, a_t, s_{t+1})\), where

- \(s_t = [\texttt{achieved\_goal},\; \texttt{observation}]\) with shape \((107,)\)
- \(g^\star = \texttt{desired\_goal}\) with shape \((2,)\)
- \(E(s_t) = s_t\) via `IdentityEncoder` (NumPy copy; not learned)

Actions have shape `(8,)`.

```bash
python scripts/inspect_antmaze.py
python scripts/collect_random_transitions.py
pytest
```

`collect_random_transitions.py` prints statistics in memory and does not write a dataset.

## Notation

| Symbol | Meaning | Status |
| --- | --- | --- |
| \(E\) | encoder | identity only |
| \(\pi_L\) | low-level worker | implemented (BC toward future x/y) |
| \(f_L\) | one-step world model | implemented (delta MLP) |
| \(\pi_H\) | high-level controller | not implemented |
| \(f_H\) | high-level model | not implemented |

Offline dataset source is not selected yet. Random rollouts stay in memory.

## Milestone 2 — one-step \(f_L\)

**Goal:** learn primitive one-step dynamics

\[
f_L(s_t, a_t) \rightarrow \hat s_{t+1}
\]

with the current representation \(E(s_t) = s_t\) (identity; 107-D baseline state).

This is **not** Director and **not** HWM. Both of those systems will reuse the same \(f_L\). RSSM and JEPA are deferred on purpose so dynamics quality is not mixed with representation learning.

The network predicts a **state delta** \(\hat s_{t+1} - s_t\) (implementation choice). Wu’s framework only requires one-step prediction, not residual form.

```bash
pip install -e ".[dev]"
python scripts/collect_random_transitions.py --n-transitions 5000 --print-state-stats
python scripts/train_low_level_dynamics.py --n-transitions 5000
pytest
```

Do not persist transitions to disk yet. Train/validation splits are **trajectory/episode-level**: adjacent transitions from the same rollout are never divided across splits. Normalization statistics are fit only on training episodes.

## Milestone 3 — shared worker \(\pi_L\)

**Goal:** implement the remaining shared low-level piece (with \(E\) and \(f_L\)):

\[
\pi_L(s_t, g_\tau) \rightarrow a_t
\]

First baseline: \(g_\tau \in \mathbb{R}^2\) is a **target future x/y** from the same episode, at most \(K\) steps ahead (default \(K=10\)). Training is behavior cloning of the recorded action \(a_t\).

This worker is **shared by Director and HWM**. There is still no high-level policy \(\pi_H\), no explicit \(f_H\), and no RSSM/JEPA.

```bash
python scripts/train_worker.py --n-transitions 5000 --horizon-k 10
pytest
```

Closed-loop eval restores the **recorded MuJoCo pose** (`qpos`/`qvel` at `s_t`), not just torso x/y. Subgoals are kept only if initial x/y distance is in `[--min-subgoal-distance, --max-subgoal-distance]` (defaults 0.5–2.0 m). Each sampled trial is rolled out **three times from the same restore**: learned `π_L`, zero torques, and uniform random actions. Compare distance reduction across those controllers; momentum can move the ant even with zeros. Judge with distance reduction and progress fractions; raw success rate is misleading if many targets already sit inside the 0.5 m radius (`fraction_already_successful_at_start`).

Random-action rollouts are a **development/debug dataset**, not the final offline corpus. They may contain few (or no) pairs 0.5–2.0 m apart within `K` steps; if eval errors, collect more/longer trajectories or widen the distance window.
