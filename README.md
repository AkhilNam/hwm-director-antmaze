# hwm-director-antmaze

Compare Director and an explicit hierarchical world model (HWM) on AntMaze.

Implemented now: identity encoder $E$, low-level worker $\pi_L$, one-step dynamics $f_L$, Director with implicit $f_H = (f_L, \pi_L)^K$, and HWM with explicit learned $f_{H\phi}$. Not implemented: RSSM, JEPA, MPC, online RL.

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

$s_t$ is the concatenation of `achieved_goal` and `observation`. $g^{\star}$ is `desired_goal`. $E$ is `IdentityEncoder` (NumPy copy, not learned).

$$
s_t \in \mathbb{R}^{29}, \quad g^{\star} \in \mathbb{R}^{2}, \quad a_t \in \mathbb{R}^{8}, \quad E(s_t) = s_t
$$

Dimensions live in `src/hwm_director/data/state.py` as `STATE_DIM = ACHIEVED_GOAL_DIM + OBSERVATION_DIM`.

The 29-D state is also enough to recover Ant-v4 simulator coordinates: `qpos = state[0:15]`, `qvel = state[15:29]` (`ant_v4_qpos_qvel_from_state`). The loader prefers recorded Minari `infos` when they are present.

## Components

| Symbol | Meaning | Status |
| --- | --- | --- |
| $E$ | encoder | identity |
| $\pi_L$ | low-level worker | behavior cloning toward future $x,y$ |
| $f_L$ | one-step world model | delta MLP |
| $\pi_H$ | high-level manager | same SoftReach $Q_H$ machinery for frozen Director and HWM |
| $f_H$ | high-level model | Director: implicit $(f_L, \pi_L)^K$; HWM: explicit $f_{H\phi}$ |
| $Q_H$ | high-level value scorer | shared; used by $\pi_H$ only, **not** $f_H$ |

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

## Current results

Minari `D4RL/antmaze/umaze-v1`, episode-level train/val split.

### One-step world model $f_L$

- Validation MSE: 0.0219
- No-change baseline MSE: 0.6163
- Validation $x,y$ MSE: 0.0033

### Goal-conditioned worker $\pi_L$

- Validation action MSE: 0.0876
- Zero-action baseline MSE: 0.5737

Closed-loop subgoal evaluation:

- Worker final distance: 0.128 m
- Zero-action final distance: 0.272 m
- Random-action final distance: 0.275 m
- Worker success rate: 100%
- Zero/random success rate: 85%

## Director

$$
\pi_H(s_t, g^{\star}) \rightarrow g_\tau
$$

$\pi_H$ maps the current 29-D state and final task goal to a local $x,y$ subgoal. The worker $\pi_L$ then acts toward that subgoal for $K$ primitive steps (default $K = 10$).

Director does not train a separate high-level world model. High-level prediction is the closed-loop composition

$$
a_i = \pi_L(s_i, g_\tau), \quad s_{i+1} = f_L(s_i, a_i)
$$

repeated $K$ times, i.e. Director $f_H = (f_L, \pi_L)^K$. The worker action is recomputed from each new predicted state.

Two rollouts:

- Real rollout: $\pi_L$ + AntMaze `env.step`
- Model rollout: $\pi_L$ + learned $f_L$ (no environment)

Manager training example: from the same episode, $(s_t, g^{\star}) \rightarrow s_{t+K}[:2]$. Examples with fewer than $K$ remaining steps are skipped. Checkpoints: `checkpoints/f_l.pt`, `checkpoints/pi_l.pt`, `checkpoints/pi_h_director.pt`.

```bash
python scripts/train_low_level_dynamics.py --save-checkpoint --checkpoint-path checkpoints/f_l.pt
python scripts/train_worker.py --horizon-k 10 --save-checkpoint --checkpoint-path checkpoints/pi_l.pt
python scripts/train_director.py --horizon-k 10 --n-env-eval-trials 3
```

Diagnostics (no retraining, no HWM):

```bash
python scripts/train_director.py --skip-manager-train \
  --n-env-eval-trials 50 \
  --n-multi-horizon-trials 20 \
  --save-eval-csv artifacts/director_eval.csv
```

This records every high-level decision, subgoal reach rate, progress to $g^{\star}$, a nearest-dataset-future $x,y$ diagnostic, stuck intervals, manager BC error, and implicit $f_H$ error at $1K,2K,3K,5K$. It does not change $\pi_H$, $\pi_L$, or $f_L$.

## Director managers

Director-BC: $\pi_H$ imitates the recorded $K$-step future $x,y$ from the same episode, $(s_t, g^{\star}) \rightarrow s_{t+K}[:2]$.

Director-Value: $\pi_H$ selects among offline-supported local subgoals using a trajectory-aware high-level value model $Q_H(h_\tau, g_\tau, g^{\star})$. $Q_H$ scores candidates; it does not predict the next high-level state.

Candidate source matching (ablation; $Q_H$, $\pi_L$, $f_L$, and $K$ are unchanged):

- Director-Value-XY: nearest offline source $x,y$, then that state's recorded $K$-step future $x,y$.
- Director-Value-State: nearest **normalized 29-D** source states (train-only `StateNormalizer`), then those recorded futures. Sources above an empirical normalized-state distance (train 90th percentile of same-episode $x,y$-nearby pairs) are dropped.
- hybrid: $\alpha$ times normalized 29-D distance plus $\beta$ times normalized $x,y$ distance (`--state-distance-weight`, `--xy-distance-weight`).
- optional hard reachability filter: reject a candidate if $\|(f_L, \pi_L)^K(s, g)[:2] - g\| > 0.5$ m. If no candidate survives, $\pi_H$ falls back to Director-BC.
- Director-Value-State-SoftReach: same state-aware candidates, **no** hard reachability cutoff. Combined score over the current candidate set:

$$
\mathrm{score} = z(Q_H) - \lambda\, z(\|(f_L, \pi_L)^K(s, g)[:2] - g\|)
$$

where $z(x) = (x - \mathrm{mean}) / (\mathrm{std} + 10^{-8})$ uses the current candidate set (not a global scaler). `--reachability-score-normalization raw` subtracts $\lambda$ times meters from raw $Q_H$ instead. $\lambda = 0$ ranks by $Q_H$ only (Value-State).

$Q_H$ is **not** $f_H$. Director high-level dynamics remain

$$
f_H^{\mathrm{Director}} = (f_L, \pi_L)^K
$$

The frozen Director comparison in this repo is **Director-SoftReach** with $\lambda = 1.0$ (50 matched seeds: success $32\%$, subgoal reach $88.1\%$). Do not retune Director for the HWM comparison.

Unsuccessful offline episodes (never within $0.5$ m of $g^{\star}$) get value target $0$. Successful episodes use $\gamma^{n}$ where $n$ is remaining $K$-step hops until first success ($\gamma = 0.99$). Episode timeout is not treated as success.

```bash
python scripts/train_director.py --skip-manager-train --skip-value-train \
  --manager-objective soft-reach --n-env-eval-trials 50 \
  --n-model-rollout-trials 0 --n-multi-horizon-trials 0 \
  --save-eval-csv artifacts/director_eval.csv
```

`--candidate-retrieval {xy,state,hybrid}` selects the source-matching mode for `--manager-objective value`. `--reachability-score-weight` enables soft scoring for a single value run. Matched evaluation uses the same $\pi_L$, $f_L$, $Q_H$, $K$, environment, and seeds. Checkpoint: `checkpoints/high_level_value.pt` (gitignored).

## Explicit HWM

HWM reuses the same $E$, $\pi_L$, $f_L$, $Q_H$, candidate retrieval, $K=10$, SoftReach $\lambda=1.0$, dataset, and evaluation seeds as frozen Director. The only intended difference is the high-level transition used for candidate reachability:

$$
f_H^{\mathrm{Director}}(h_\tau, g_\tau) = (f_L, \pi_L)^K(h_\tau, g_\tau)
$$

$$
f_H^{\mathrm{HWM}}(h_\tau, g_\tau) = f_{H\phi}(h_\tau, g_\tau)
$$

$f_{H\phi}$ is a delta MLP $31 \rightarrow 256 \rightarrow 256 \rightarrow 29$:

$$
\hat{h}_{\tau+1} = h_\tau + \mathrm{MLP}([h_\tau, g_\tau])
$$

Training examples are recorded $K$-step tuples from the same episode: $(s_t, s_{t+K}[:2]) \rightarrow s_{t+K}$. That is supervised on dataset behavior. It is **not** a closed-loop rollout of the current learned $\pi_L$.

Normalization (train-episode states only):

$$
h_{\tau}^{\mathrm{n}} = \mathrm{normalize}(h_\tau),\quad
g_{\tau}^{\mathrm{n}} = \mathrm{normalize}(g_\tau[:2]),\quad
h_{\tau+1}^{\mathrm{n}} = \mathrm{normalize}(h_{\tau+1})
$$

$$
\Delta^{\mathrm{n}} = h_{\tau+1}^{\mathrm{n}} - h_{\tau}^{\mathrm{n}}
$$

Loss is MSE between $h_{\tau}^{\mathrm{n}} + \widehat{\Delta}^{\mathrm{n}}$ and $h_{\tau+1}^{\mathrm{n}}$.

At test time HWM still executes the shared $\pi_L$ in real AntMaze. $f_{H\phi}$ is used only to score candidate reachability (SoftReach) and for model-quality comparisons. Imagined $f_{H\phi}$ states are never stepped in the simulator.

Checkpoint: `checkpoints/f_h_explicit.pt` (gitignored).

Matched 50-seed evaluation (same seeds, $Q_H$, candidates, $\lambda=1$, $K=10$):

| | Director $(f_L,\pi_L)^K$ | HWM $f_{H\phi}$ |
| --- | --- | --- |
| success | $32\%$ | $22\%$ |
| subgoal reach | $88.1\%$ | $74.5\%$ |
| stuck among failed | $100\%$ | $100\%$ |
| mean final distance | $1.31$ m | $1.49$ m |
| U-wall steps | $22.3$ | $30.2$ |
| pred. vs actual reach Pearson | $0.82$ | $0.41$ |

$f_{H\phi}$ validation: full-state MSE $0.539$, $x,y$ error $0.080$ m (no-change $x,y$ $0.366$ m). On matched 1K starts, HWM $x,y$ error vs real $\pi_L$ is $0.118$ m vs Director $0.144$ m vs no-change $0.365$ m. Recursive $1K$–$5K$ error is a different protocol than this one-step number.

```bash
python scripts/train_hwm.py --n-env-eval-trials 50 \
  --n-model-rollout-trials 20 --n-multi-horizon-trials 20 \
  --save-eval-csv artifacts/director_eval.csv
python scripts/train_hwm.py --skip-high-level-dynamics-train --n-env-eval-trials 50
```

## Scripts

```bash
python scripts/inspect_minari_antmaze.py
python scripts/inspect_antmaze.py
python scripts/train_low_level_dynamics.py
python scripts/train_worker.py --horizon-k 10
python scripts/train_director.py --horizon-k 10
python scripts/train_director.py --skip-manager-train --skip-value-train --manager-objective soft-reach --n-env-eval-trials 50
python scripts/train_hwm.py --n-env-eval-trials 50
pytest
```

`scripts/collect_random_transitions.py` is a smoke-test utility, not the training data source.
