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
| \(\pi_L\) | low-level worker | not implemented |
| \(f_L\) | one-step world model | not implemented |
| \(\pi_H\) | high-level controller | not implemented |
| \(f_H\) | high-level model | not implemented |

Offline dataset source is not selected yet.
