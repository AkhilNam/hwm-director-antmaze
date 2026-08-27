# hwm-director-antmaze

Compare Director and an explicit hierarchical world model (HWM) on AntMaze.

Implemented now: identity encoder $E$, low-level worker $\pi_L$, and one-step dynamics $f_L$. Not implemented: $\pi_H$, $f_H$, RSSM, JEPA, Director, HWM.

## Setup

Python 3.10–3.13 (Gymnasium-Robotics). Prefer 3.12 or 3.13 if the system Python is newer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Dataset and environment

| | |
| --- | --- |
| Offline dataset | Minari [`D4RL/antmaze/umaze-v1`](https://minari.farama.org/datasets/D4RL/antmaze/umaze-v1/) |
| Environment | `AntMaze_UMaze-v4` (recovered from the Minari dataset when possible) |
| Package | [`gymnasium-robotics==1.4.2`](https://pypi.org/project/gymnasium-robotics/) |
| API | Gymnasium (`observation, reward, terminated, truncated, info`) |
| Physics | official `mujoco` Python bindings (not `mujoco-py`) |

Raw observations are a dict:

| Key | Shape | Meaning |
| --- | --- | --- |
| `observation` | `(27,)` | Ant-v4 proprioception (no torso $x,y$) |
| `achieved_goal` | `(2,)` | current torso $x,y$ |
| `desired_goal` | `(2,)` | task target $x,y$ |

Actions have shape `(8,)`.

Minari episode `infos` include `qpos` `(15,)` and `qvel` `(14,)` at each observation. Those arrays are used to restore MuJoCo state for closed-loop worker evaluation.

## Representation

$$
s_t = [\texttt{achieved\_goal},\; \texttt{observation}] \in \mathbb{R}^{29}
$$

$$
g^\star = \texttt{desired\_goal} \in \mathbb{R}^{2}, \qquad a_t \in \mathbb{R}^{8}
$$

$$
E(s_t) = s_t
$$

(`IdentityEncoder`: NumPy copy, not learned.) Dimensions live in `src/hwm_director/data/state.py` as `STATE_DIM = ACHIEVED_GOAL_DIM + OBSERVATION_DIM`.

The 29-D state is also enough to recover Ant-v4 simulator coordinates: `qpos = state[0:15]`, `qvel = state[15:29]` (`ant_v4_qpos_qvel_from_state`). The loader prefers recorded Minari `infos` when they are present.

## Components

| Symbol | Meaning | Status |
| --- | --- | --- |
| $E$ | encoder | identity |
| $\pi_L$ | low-level worker | behavior cloning toward future $x,y$ |
| $f_L$ | one-step world model | delta MLP |
| $\pi_H$ | high-level controller | not implemented |
| $f_H$ | high-level model | not implemented |

Train/validation splits are episode-level. Normalization statistics are fit on training episodes only.

## Low-level dynamics

$$
f_L(s_t, a_t) \rightarrow \hat{s}_{t+1}
$$

The network predicts a state delta; the next state is reconstructed as $\hat{s}_{t+1} = s_t + \widehat{\Delta}_t$.

```bash
python scripts/train_low_level_dynamics.py
python scripts/train_low_level_dynamics.py --max-episodes 8 --epochs 5
```

## Worker

$$
\pi_L(s_t, g_\tau) \rightarrow a_t
$$

$g_\tau \in \mathbb{R}^{2}$ is a future torso $x,y$ from the same episode, at most $K$ steps ahead (default $K = 10$). Training is behavior cloning of the recorded action $a_t$. Network input dimension is $29 + 2 = 31$.

Closed-loop evaluation restores recorded `qpos`/`qvel` at $s_t$. Eligible subgoals have initial $x,y$ distance in `[--min-subgoal-distance, --max-subgoal-distance]` (defaults $0.5$–$2.0$ m). Each trial is rolled out three times from the same restore: learned $\pi_L$, zero torques, and uniform random actions. If `qpos`/`qvel` are missing, that evaluation path is skipped; behavior-cloning metrics are still reported.

```bash
python scripts/train_worker.py --horizon-k 10
python scripts/train_worker.py --max-episodes 8 --epochs 5 --n-eval-trials 0
```

## Scripts

```bash
python scripts/inspect_minari_antmaze.py
python scripts/inspect_antmaze.py
python scripts/train_low_level_dynamics.py
python scripts/train_worker.py --horizon-k 10
pytest
```

`scripts/collect_random_transitions.py` is a smoke-test utility, not the training data source.
