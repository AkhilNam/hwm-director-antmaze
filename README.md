# hwm-director-antmaze

Director vs explicit hierarchical world model (HWM) on AntMaze. Shared: identity $E$, worker $\pi_L$, one-step $f_L$. Director $f_H = (f_L, \pi_L)^K$. HWM $f_H$ is a learned $f_{H\phi}$. Not implemented: RSSM, JEPA, MPC, online RL.

## Setup

Python 3.10–3.13. Prefer 3.12 or 3.13 if the system Python is newer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Data

Minari [`D4RL/antmaze/umaze-v1`](https://minari.farama.org/datasets/D4RL/antmaze/umaze-v1/), env `AntMaze_UMaze-v4`. State $s_t \in \mathbb{R}^{29}$ is `achieved_goal` $(2,)$ plus Ant proprioception $(27,)$. Goal $g^{\star}$ is `desired_goal` $(2,)$. Action $(8,)$. $E(s_t)=s_t$. $K=10$. Episode-level train/val split; normalizers are fit on train episodes only.

## Hierarchy

| | | |
| --- | --- | --- |
| $\pi_L$ | worker | BC toward a future $x,y$ |
| $f_L$ | one-step model | $\hat{s}_{t+1} = s_t + \mathrm{MLP}([s_t, a_t])$ |
| $\pi_H$ | manager | SoftReach over data-supported subgoals using shared $Q_H$ |
| $f_H$ | high-level model | Director: $(f_L, \pi_L)^K$; HWM: $f_{H\phi}(h_\tau, g_\tau) \to \hat{h}_{\tau+1}$ |
| $Q_H$ | scorer | ranks candidates; not a dynamics model |

Every $K$ steps, $\pi_H$ picks a local $x,y$ subgoal $g_\tau$. Real execution always uses $\pi_L$ in AntMaze. SoftReach score (candidate-set $z$-scores, $\lambda=1$):

$$
\mathrm{score} = z(Q_H) - \lambda\, z(\|\hat{h}_{\tau+1}[:2] - g_\tau\|)
$$

Director and HWM share $E$, $\pi_L$, $f_L$, $Q_H$, candidates, $K$, $\lambda$, dataset, and eval seeds. They differ only in $\hat{h}_{\tau+1}$. $f_{H\phi}$ is trained on recorded $(s_t, s_{t+K}[:2]) \to s_{t+K}$, not a rollout of the current $\pi_L$.

## Results

`D4RL/antmaze/umaze-v1`.

| | val | baseline |
| --- | --- | --- |
| $f_L$ MSE | 0.0219 | no-change 0.6163 |
| $f_L$ $x,y$ MSE | 0.0033 | |
| $\pi_L$ action MSE | 0.0876 | zero-action 0.5737 |
| $\pi_L$ closed-loop distance | 0.128 m | zero 0.272 m, random 0.275 m |

50 matched seeds, SoftReach $\lambda=1$:

| | Director | HWM |
| --- | --- | --- |
| success | 32% | 22% |
| subgoal reach | 88.1% | 74.5% |
| mean final distance | 1.31 m | 1.49 m |
| pred. vs actual reach Pearson | 0.82 | 0.41 |
| $f_H$ 1K $x,y$ vs real $\pi_L$ | 0.144 m | 0.118 m |

$f_{H\phi}$ val MSE 0.539, $x,y$ error 0.080 m (no-change 0.366 m).

## Commands

```bash
python scripts/train_low_level_dynamics.py --save-checkpoint --checkpoint-path checkpoints/f_l.pt
python scripts/train_worker.py --horizon-k 10 --save-checkpoint --checkpoint-path checkpoints/pi_l.pt
python scripts/train_director.py --horizon-k 10 --n-env-eval-trials 50
python scripts/train_hwm.py --n-env-eval-trials 50
pytest
```

Checkpoints (gitignored): `checkpoints/f_l.pt`, `pi_l.pt`, `pi_h_director.pt`, `high_level_value.pt`, `f_h_explicit.pt`.
