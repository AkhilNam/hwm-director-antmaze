# hwm-director-antmaze

Research codebase for a future comparison of **Director** and an explicit **Hierarchical World Model (HWM)** on AntMaze.

**Milestone 1 is environment inspection only.** There are no learned models, training loops, or datasets in this repository yet.

## Environment

- Package: [`gymnasium-robotics==1.4.2`](https://pypi.org/project/gymnasium-robotics/)
- Env ID: `AntMaze_UMaze-v5`
- API: Gymnasium (`observation, reward, terminated, truncated, info`)
- Physics: official `mujoco` Python bindings (not `mujoco-py` / D4RL)

Gymnasium-Robotics currently documents Python 3.10–3.13. Use a 3.12 or 3.13 virtualenv if the system Python is newer.

## Install and inspect

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/inspect_antmaze.py
```

## Future notation (not implemented)

These names are reserved for later milestones. **None of them are implemented yet.**

| Symbol | Meaning |
| --- | --- |
| \(E\) | encoder (observation / history → latent state) |
| \(\pi_L\) | low-level worker |
| \(f_L\) | one-step (low-level) world model |
| \(\pi_H\) | high-level controller (Director manager or planner) |
| \(f_H\) | high-level model (explicit in HWM; implicit / absent in Director) |

Also not in this milestone: PyTorch, RSSM, JEPA, hierarchical rollouts, offline RL, or rendering.

## Offline data

**The offline dataset source has not been selected yet.** Do not assume D4RL, Minari, or any other corpus until that decision is recorded here.
