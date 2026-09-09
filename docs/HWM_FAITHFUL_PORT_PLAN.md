# Faithful HWM Port Plan

**Status:** Architecture plan complete. **M1 implemented** (Level-1 observation encoder). **M2 implemented** (Level-1 residual conv predictor + recursive rollout, shape/architecture parity only). M3 and later are not started. No training, no dataset regeneration, no PACE jobs.

**Date:** 2026-09-09 (M1: 2026-09-09; M2: 2026-09-09)

**Original HWM reference:** [kevinghst/HWM_PLDM](https://github.com/kevinghst/HWM_PLDM) at SHA `e197375b844692a0a2e1342889f95a78edced07a` (read-only). Local clone used for tracing: `hwm-original-repro`.

**Unified repo:** [AkhilNam/hwm-director-antmaze](https://github.com/AkhilNam/hwm-director-antmaze). Existing simplified Director/HWM is the **raw-state baseline** and must remain reproducible.

**First validation target:** original Diverse Maze hierarchical planning, not AntMaze.

**Reference original-code reproduction (already obtained, do not overwrite):**

| Split | Our original-code result | Paper (approx.) |
| --- | --- | --- |
| medium | 82.5% | ~95% |
| hard | 90.0% | ~83% |

Source of our numbers: original eval summary `l2_d4rl_medium_planning_success_rate = 0.825`, `l2_d4rl_hard_planning_success_rate = 0.9`.

---

## 0. How to read this document

Original HWM is **not** a raw-state MLP plus a candidate-Q manager. It is a two-level Joint-Embedding Predictive Architecture (HJEPA / PLDM) with:

1. a **learned visual representation** (Level-1 encoder),
2. a **Level-1 residual conv predictor** over primitive 2D actions,
3. a **Level-2 residual conv predictor** over inferred latent macro-actions `z`,
4. **hierarchical MPPI** (no learned `pi_H` network).

Professor Wu's abstraction `M_hier = (pi_H, f_H, pi_L, f_L)` is still the right scientific language, but the original HWM mapping is imperfect. The imperfect parts are called out in Section 7 rather than forced.

Naming in *our* repo going forward:

| Name | Meaning | Code location |
| --- | --- | --- |
| `HWMRawState` | Current simplified HWM (identity `E`, 29D AntMaze, MLP `f_H`, candidate/`Q_H`) | existing `src/hwm_director/` — **do not replace** |
| `HWMFaithful` | New independently implemented HWM targeting original Diverse Maze | proposed `src/hwm_faithful/` |

---

## 1. End-to-end original HWM dataflow

### 1.1 Two-stage training (verified from YAML + `train.py`)

The ICML Diverse Maze system is trained in **two stages**, not jointly.

**Stage A — Level 1 only** (`pldm/configs/diverse_maze/icml/large_diverse_25maps.yaml`):

- `hjepa.disable_l2: true`
- `hjepa.train_l1: true`
- `hjepa.freeze_l1: false`
- Objectives: `VICRegObs`, `IDM`, `PredictionProprio`
- Data: `maze2d_large_diverse_25maps` images + `data.p`
- Sequence length `n_steps = 15`

**Stage B — Level 2 on frozen Level 1** (`large_diverse_25maps_l2.yaml`):

- `train_l1: false`, `freeze_l1: true`, `load_l1_only: true`
- Loads a pretrained L1 checkpoint
- `disable_l2: false`
- Objectives: `PredictionObs`, `PredictionProprio`
- VICReg coefficients appear in the YAML but are **not** in `objectives_l2.objectives` (unused for L2)
- L2 sequence: `l2_n_steps = 6`, `l2_step_skip = 10`

This matches the checkpoint name used in reproduction: `load_from_l1248-seed248_epoch=5_sample_step=10789632.ckpt`.

### 1.2 Observation pipeline

Point-mass maze (`maze2d_large_diverse`), **not** Ant.

- Simulator state: `(x, y, vx, vy)` in `R^4`
- Images: top-down RGB, **98×98**, rendered from `(x, y)`
- Primitive action: 2D (`R^2`), typically acceleration/force in `[-1, 1]` after bounding
- Dataset generation and eval both use `action_repeat = 4`, `action_repeat_mode = 'id'`
  - one dataset / planner step = four identical physics steps
- Offline training data: 25 maze layouts, 2000 episodes/layout, episode length 100 wrapper steps (`pldm_envs/diverse_maze/configs/maze2d_large/25maps.yaml`)
- HuggingFace snapshot used by the authors: `kevinghst/maze2d-large-diverse-25maps`

### 1.3 Forward pass at training time (`HJEPA.forward_posterior`)

For the L2 config (`train_l1 = false`):

1. Dataset returns **subsampled** image sequence `l2_states` and **chunked** primitive actions `l2_actions`.
2. Frozen L1 backbone encodes each subsampled image + proprio vel → spatial representation.
3. L2 backbone is `identity_encoder`: it concatenates L1 `obs_component` and L1 `proprio_component`. It does **not** mean the whole HWM is identity/raw-state.
4. For each L2 step, a posterior MLP maps the 10-step primitive-action chunk → latent `z ∈ R^8` (deterministic mean).
5. L2 conv predictor residual-predicts the next L1 representation, conditioned on `z`.
6. Losses compare predicted vs encoded L2 `obs_component` and `proprio_component`.

### 1.4 Forward pass at planning time

No posterior. MPPI proposes `z` sequences (L2) and primitive action sequences (L1). Predictors are rolled out open-loop in representation space. The environment is queried only for the first primitive action of the L1 plan, then the real image is re-encoded.

### 1.5 Evaluation

`Evaluator.evaluate()`:

1. Train location **probers** (linear/conv maps from `obs_component` → `(x, y)`). Used for plots, not for the success bit.
2. If `eval_l2` and not `disable_l2_planning`: run `HierarchicalD4RLMPCEvaluator` on medium and hard start/goal files.
3. Success = binary env reward = `||xy - target_xy||_2 < 0.5`.

The L2 YAML sets `disable_planning: true` (skips **flat L1-only** planning) and `disable_l2_planning: false` (runs hierarchical planning).

---

## 2. Shape-level walkthrough

Shapes below are traced from:

- encoder config `d4rl_a` + 98×98 input + conv arithmetic
- `MeNet6`, `IdentityEncoder`, `ConvPredictor`, `SequencePredictor`
- `D4RLDataset.__getitem__`
- `JEPA` posterior_input_dim override
- `TwoLvlPlanner` / `MPPIPlanner` / `LearnedDynamics`

`B` is batch (training) or number of parallel envs (planning). Time is leading in the model (`T × B × ...`) after `transpose(0, 1)` in `train.py`.

### 2.1 Level-1 encoder

```
image I_t                          [B, 3, 98, 98]
proprio vel v_t = (vx, vy)         [B, 2]

late proprio expander (id_expand):
  v_t tiled spatially              [B, 2, 43, 43]

visual conv (d4rl_a, 5 layers):
  98 → 94 → 45 → 43 → 43 → 43
obs_component o_t                  [B, 16, 43, 43]
proprio_component p_t              [B,  2, 43, 43]
encodings h_t = concat(o_t, p_t)   [B, 18, 43, 43]   ← fused L1 state
```

`backbone_width_factor: 2` is present in YAML and in `BackboneConfig`, but **no Python code reads it**. Actual channels are those in `ENCODER_LAYERS_CONFIG["d4rl_a"]`.

L1 `input_dim: 4` in YAML is also unused; `Trainer` infers `input_dim` from the dataset tensor (`[3, 98, 98]`).

### 2.2 Level-1 predictor (`conv2` / `d4rl_b_p`)

```
h_t                                [B, 18, 43, 43]
a_t primitive action               [B, 2]
action expander (arch = id):       [B,  2, 43, 43]
concat                             [B, 20, 43, 43]
conv residual → ĥ_{t+1}            [B, 18, 43, 43]
  split last 2 channels as proprio
predicted obs                      [B, 16, 43, 43]
predicted proprio                  [B,  2, 43, 43]
```

`z_dim = 0` at Level 1. The predictor is conditioned on primitive actions only.

Training sequence (`n_steps = 15`):

```
images                             [15, B, 3, 98, 98]
actions                            [14, B, 2]
encoded h                          [15, B, 18, 43, 43]
predictions (incl. t=0 copy)       [15, B, 18, 43, 43]
loss uses timesteps 1..14 vs encoded 1..14
```

### 2.3 Level-2 input and latent `z`

`l2_n_steps = 6`, `l2_step_skip = 10`:

```
wrapper-time window length         = 6 * 10 + 1 = 61

l2_states  (images, skip 10)       [7, B, 3, 98, 98]
l2_proprio_vel (skip 10)           [7, B, 2]
l2_locations (xy, skip 10)         [7, B, 2]

primitive actions over 60 steps    [60, B, 2]
chunked l2_actions                 [6, B, 10, 2]
flattened posterior input          [6, B, 20]     # 10 * action_dim 2

L1 encode_only on l2_states:
  o_τ                              [7, B, 16, 43, 43]
  p_τ                              [7, B,  2, 43, 43]

L2 identity_encoder:
  encodings H_τ = concat(o_τ, p_τ) [7, B, 18, 43, 43]
```

Posterior (`posterior_arch = 32-32`, `posterior_input_type = actions`):

```
MLP: 20 → 32 → 32 → 16, then chunk into (mu, std)
LayerNorm on mu
std = softplus(std) + z_min_std (0.05)

Because z_stochastic = false:
  z_τ = mu                         [B, 8]
```

`prior_arch = ""` → dummy prior `N(0, I)`. The L2 YAML does **not** include a KL objective, so the prior is unused at train time except for logging tensors.

### 2.4 Level-2 predictor (`conv2` / `l2_d4rl_e_p`)

```
H_τ                                [B, 18, 43, 43]
z_τ                                [B, 8]
action encoder 8-64-8 + expand     [B, 8, 43, 43]
concat                             [B, 26, 43, 43]
  (YAML table starts at 42 in-channels; build_conv overrides layer 0
   with the computed 26)
5-layer conv, residual on H_τ
Ĥ_{τ+1}                            [B, 18, 43, 43]
predicted L2 obs                   [B, 16, 43, 43]
predicted L2 proprio               [B,  2, 43, 43]
```

Open-loop L2 rollout of horizon `T2`:

```
predicted trajectory               [T2+1, B, 18, 43, 43]
```

### 2.5 Planning tensors

**Level-2 MPPI** (latent actions, `action_dim = 0`, `z_dim = 8`):

```
candidate z sequences              [K2, T2, 8]
  medium: K2 = 2000, T2 ≤ 35
  hard:   K2 = 4000, T2 ≤ 47
predicted H trajectory             [T2+1, K2, 18, 43, 43]
cost on flattened obs_component    [K2]
chosen z plan                      [T2, 8] per env
L1 subgoal = pred_obs[1]           [B, 16, 43, 43]
  (first L2-predicted visual map, one macro-step ahead)
```

**Level-1 MPPI** (primitive actions):

```
candidate a sequences              [K1, T1, 2]
  hierarchical: T1 = l2_step_skip = 10
  medium K1 = 500; hard K1 = 1000
  flat L1-only (not used in L2 yaml): T1 ≤ 128
predicted h trajectory             [T1+1, K1, 18, 43, 43]
chosen primitive plan              [T1, 2] per env
executed action                    [2]   (first remaining action)
```

After `unnormalize_action`, the env receives a 2D maze action, **not** an 8D Ant torque.

### 2.6 Goal representation

The goal is **not** raw `(x, y)` inside the planner.

```
render goal image at target xy, vel = 0
  I*                               [B, 3, 98, 98]
L1 encode (proprio target = zeros)
  o*                               [B, 16, 43, 43]
L2 identity concat
  H* flattened obs                 [B, 16 * 43 * 43] = [B, 29584]
```

MPPI cost is MSE in this representation (`cost_entity = obs_component`, `repr_target = true`).

A separately trained **location prober** maps `obs_component → (x, y)` for logging/plots only.

---

## 3. Level-1 training objective

Config: `large_diverse_25maps.yaml` → `objectives_l1`.

Let `o_t` be L1 `obs_component`, `p_t` proprio component, `ĥ` the residual conv prediction, `a_t` primitive action.

**Sample.** Uniform windows of `n_steps = 15` consecutive wrapper steps from `maze2d_large_diverse_25maps`. Images `[15, B, 3, 98, 98]`, actions `[14, B, 2]`, proprio vel `[15, B, 2]`. Channel-wise image / action / proprio normalization (`normalizer_hardset` uses fixed RGB/action/vel stats in `pldm_envs/utils/normalizer.py`).

**Predictor.** Open-loop residual rollout for 14 steps, teacher-encoded starts from `h_0`, actions `a_0..a_13`. JEPA-style: encode the full window with the backbone, predict forward from `h_0`.

**Losses (all used):**

1. **VICRegObs** (`sim_coeff = 1`, `std_coeff ≈ 29.4`, `cov_coeff ≈ 17.9`, `std_coeff_t ≈ 2.92`)
   - `sim_loss` is MSE(`o_{t+1}`, `ô_{t+1}`) — this **is** the visual prediction term
   - variance/covariance regularizers on flattened `o` (and a temporal std term)
2. **IDM** (`coeff ≈ 4.81`, conv subclass `a`)
   - predict `a_t` from concat(`h_t`, `h_{t+1}`)
3. **PredictionProprio** (`global_coeff ≈ 2.42`)
   - MSE(`p_{t+1}`, `p̂_{t+1}`)

There is **no** separate `PredictionObs` entry; visual prediction is the VICReg invariance/sim term.

L1 is **not** trained with a latent `z`.

---

## 4. Level-2 training objective

Config: `large_diverse_25maps_l2.yaml` → `objectives_l2`. L1 frozen.

**Sample.** Same episodes. For each index:

- images every 10 wrapper steps, length 7 (`τ = 0..6`)
- 60 consecutive primitive actions, split into 6 chunks of 10

`step_skip = 10` means: L2's one step corresponds to 10 dataset / wrapper steps (40 physics steps given `action_repeat = 4`).

Macro horizon in training: **6 L2 steps**.

**What `z` is.** A deterministic 8D code inferred from the **chunk of 10 primitive actions** that actually occurred between `τ` and `τ+1`. It is an *action abstraction*, not a state. Posterior input type is `actions`, not `term_states`.

**Predictor.** Residual conv, conditioned on `z_τ` (not on primitive `a`). Because `action_dim = 0` and `z_dim = 8`, primitive actions never enter the L2 predictor directly.

**Losses actually optimized:**

```
L = 2.416 * MSE(o_{τ+1}, ô_{τ+1}) + 2.416 * MSE(p_{τ+1}, p̂_{τ+1})
```

YAML also lists VICReg hyperparameters and a probe block. Those are **not** in `objectives:`. KL is also absent. So L2 training in this config is **pure representation prediction**, with `z` coming from the action posterior.

---

## 5. Exact role of latent `z`

`z ∈ R^8` is the **Level-2 action**, i.e. the high-level control variable.

| Mode | How `z` is obtained | Used as |
| --- | --- | --- |
| L2 training | `z = LayerNorm(MLP_32-32(flatten(a_{t:t+10}))).mu` | conditioner of `f_H` |
| L2 planning | MPPI samples `z_{0:T2-1}` around a nominal trajectory, `N(0, 10^2 I)` by default | open-loop macro actions |
| L1 | `z` is not used | primitive `a ∈ R^2` instead |

Properties from code:

- `z_stochastic = false` → always the mean, never a reparameterized sample
- `z_discrete = false`
- `z_min_std = 0.05` still parameterizes a std head, but the std is unused for sampling
- Mutual exclusion in `JEPA`: a level has either `action_dim > 0` or `z_dim > 0`, not both
- At planning time `LearnedDynamics` calls `forward_multiple(..., latents=action)` when `model.config.action_dim == 0`
- `z_reg_coeff = 0.1` is passed into MPPI. The torch MPPI **computes** a Normal(0,1) NLL regularizer on latent actions, but in the current `mppi_torch.py` that tensor is **not added** to `cost_total`. Treat this as an original-code ambiguity (Section 16).

`z` is the closest original-HWM object to a high-level action `g_τ` in Wu's notation, except it is **not** an x/y subgoal. The x/y subgoal is only an *emergent* decoded location of the first L2-predicted visual map.

---

## 6. Hierarchical planning flow

Code path: `MPCEvaluator._perform_h_mpc` → `_perform_mpc(..., bilevel_planning=True)` → `TwoLvlPlanner.plan` → env `step`.

### 6.1 One replan cycle (plain English)

1. **Encode the current real observation.** Render 98×98 RGB, normalize, run L1 backbone with current proprio vel. This yields `o_t, p_t, h_t`.
2. **Level-2 current state.** For this checkpoint L2 backbone is identity, so `H_t = concat(o_t, p_t)`. (TwoLvlPlanner even skips calling L2 backbone and passes the L1 `BackboneOutput` into the L2 planner with `repr_input=True`.)
3. **Level-2 MPPI proposes a sequence of latent macro-actions** `z_0, z_1, ..., z_{T2-1}` in `R^8`. Medium: up to 35 steps, 2000 samples, `noise_sigma = 10`, `lambda = 0.0025`.
4. **Level-2 world model** residual-rolls `H_{τ+1} = f_H(H_τ, z_τ)` in representation space. Cost is MSE between predicted `obs_component` (flattened) and the **goal image encoding**. `sum_all_diffs: true` → all future L2 steps contribute.
5. **Best L2 plan** is the MPPI importance-weighted update of the nominal `z` sequence (Williams et al. 2017).
6. **Level-1 target** is `pred_obs[1]`: the *first* L2-predicted visual map (one skip of 10 wrapper steps), not the whole L2 trajectory and not raw xy.
7. **Level-1 MPPI** optimizes a length-10 primitive action sequence so that L1 predicted visual features match that L2 subgoal (`cost_entity = obs_component`). Medium: 500 samples, `noise_sigma = 5`.
8. **Primitive action sent to the env** is the first remaining L1 action, unnormalized, shape `(2,)`.
9. **Env steps.** `ActionRepeatWrapper` applies that action 4 times (`mode = id`). One planner loop iteration = 4 physics steps.
10. **Replan.** `replan_every = 4` wrapper steps. Between replans the code slices `actions[:, i % 4:]` and still executes index 0 of that slice, i.e. L1 actions 0,1,2,3 of the last plan. Then re-encode the real image and repeat.

### 6.2 Mathematics (one hierarchical step)

Let `E_1` be the L1 encoder, `f_1` the L1 predictor, `f_2` the L2 predictor, `φ` flatten of `obs_component`.

Goal encoding:

```
H* = E_2(E_1(I*, v*=0))
c_2(z_{0:T2}) = Σ_{k=1}^{T2} || φ(f_2^k(H_t, z_{0:k-1})) - φ(H*) ||^2
ẑ = MPPI(c_2)
```

Subgoal for L1:

```
o_sg = obs_component( f_2(H_t, ẑ_0) )     # pred_obs[1]
```

L1 plan of length `K=10`:

```
c_1(a_{0:9}) = Σ_{j} || φ(f_1^j(h_t, a_{0:j-1})) - φ(o_sg) ||^2
â = MPPI(c_1)
env ← â_0   (repeated 4 physics steps)
```

### 6.3 Horizons and the two-stage MPC wrapper

For **medium** (`n_steps = 500`, `max_plan_length_l2 = 35`, `final_trans_steps = 15`, `step_skip = 10`):

```
max_plan_horizon_l2 = 35 * 10 = 350 wrapper steps
stage-1 hierarchical max_steps = min(500 - 15, 350) = 350
stage-2 flat L1 MPC for 15 more wrapper steps
total wrapper steps used ≈ 365  (500 is not fully consumed)
```

Hard: `max_plan_length_l2 = 47` → stage-1 cap `470`, plus 15 flat L1.

`final_trans_steps` exists because the authors found hierarchical MPC “might have some issue with fine-grained control” near the goal (`mpc.py` comment). Stage 2 uses **only** `f_L` / L1 MPPI toward the encoded goal.

`probe_depth: false` in the L2 YAML, so `determine_optimal_depths` is not used in the default eval.

### 6.4 Success

`CustomMazeEnv._is_goal_reached(goal_dist_threshold=0.5)`:

```
|| xy_t - xy* ||_2 < 0.5  ⇒  reward = 1, else 0
```

Report success rate = fraction of envs for which some `reward_history` entry is truthy before timeout. `n_envs = 40` per difficulty. Starts/targets from `starts_targets_9_12.pt` (medium) and `starts_targets_13_16.pt` (hard).

---

## 7. Mapping into `M_hier = (pi_H, f_H, pi_L, f_L)`

Do **not** pretend this is a clean 1-1 with the current AntMaze code.

| Wu symbol | Original HWM | Notes |
| --- | --- | --- |
| `ρ` / `E` | L1 visual encoder `MeNet6` (`d4rl_a`) + late proprio expander | Learned. Output is spatial `[16+2, 43, 43]`, not 29D identity. |
| `f_L` | L1 residual conv predictor, primitive `a ∈ R^2` | Predictive model in **representation** space, not `s_{t+1} = s_t + MLP([s,a])`. |
| `pi_L` | **L1 MPPI planner**, not a BC worker | Optimizes primitive actions online with `f_L`. No cloned neural `pi_L`. |
| `f_H` | L2 residual conv predictor on `z ∈ R^8` | Explicit learned high-level world model. Operates on L1 features, skip 10. |
| `pi_H` | **L2 MPPI planner** | Not a learned manager, not candidate retrieval, not `Q_H`. |

Imperfect fits (say so rather than force them):

1. Original HWM has **no learned closed-loop worker**. Real actions always come from MPPI + `f_L`, never from `a = pi_L(s, g_xy)`.
2. High-level action is **latent `z`**, not a 2D subgoal. The L1 planner's target is a **predicted visual feature map**.
3. `f_H` does not take `(h, g_xy)`; it takes `(H, z)`.
4. There is a **second** `pi_L`-like object in stage-2 MPC (flat L1 MPPI to the **final goal encoding**, not to an L2 subgoal).
5. L2 `identity_encoder` is identity **on L1 features**, not identity on raw state.
6. Original HWM does not share a neural `pi_L` with Director. A later unified comparison will have to decide whether “faithful HWM on AntMaze” keeps MPPI as `pi_L` or replaces it with a worker.

**One-sentence mapping that is still honest:**

> Original HWM is hierarchical MPC on two learned predictive models: `pi_H = MPPI(f_H)`, `f_H = L2 JEPA`, `pi_L = MPPI(f_L)`, `f_L = L1 JEPA`, `E = L1 visual encoder`.

---

## 8. Gap table vs current unified implementation

Current repo (`src/hwm_director/`): identity `E`, 29D AntMaze, BC worker, one-step MLP `f_L`, K=10, explicit MLP `f_H` on `(s, g_xy)`, SoftReach + `Q_H` over offline candidates. Results to preserve: Director ~32% success, HWM ~22% on AntMaze umaze.

| # | Original HWM component | Purpose | Original file / class | Input shape | Output shape | Training objective | Our current equivalent | Status | Proposed implementation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Visual observation encoder | `ρ`: RGB → spatial features | `encoders.py::MeNet6`, subclass `d4rl_a` | `[B,3,98,98]` | `[B,16,43,43]` | L1 VICReg/IDM/pred (encoder trained with L1) | `IdentityEncoder`, `E(s)=s` | **MISSING** | Clean `VisualEncoder` in `hwm_faithful` |
| 2 | L1 learned representation | JEPA state `h = concat(o,p)` | `jepa.py::JEPA` + late proprio fuse | image + `[B,2]` vel | `[B,18,43,43]` | VICRegObs + IDM + PredProprio | 29D raw state | **MISSING** | L1 encoder module; keep identity encoder in baseline |
| 3 | L1 predictive WM `f_L` | `ĥ_{t+1}=h_t+Conv([h_t,a_t])` | `conv_predictors.py::ConvPredictor` `d4rl_b_p` | `[B,18,43,43]`, `[B,2]` | `[B,18,43,43]` | visual sim + proprio MSE | MLP residual on 29D+8D action | **MISSING** (different domain and architecture) | `Level1WorldModel` conv residual |
| 4 | Proprioception | vel fused as 2 extra channels | `MeNet6` late `id_expand`; maze2d has empty qpos | `[B,2]` | `[B,2,43,43]` | PredictionProprio | packed into 29D vector, no separate stream | **MISSING** as a separate stream | `ProprioExpander` |
| 5 | L2 input representation | L1 features, not raw pixels | `encoders.py::IdentityEncoder` (`arch=identity_encoder`) | L1 o,p | concat `[B,18,43,43]` | none (identity) | n/a (we have no L1 features) | **MISSING** | `Level2IdentityBackbone` with an explicit comment that this is identity-on-L1 |
| 6 | Temporal abstraction | L2 step = 10 wrapper steps | `d4rl.py` `skip_frame=l2_step_skip`; `HJEPA.step_skip` | length-61 window | 7 L2 states, 6 chunks | — | `K=10` is a worker horizon, not a learned skip | **PARTIAL** | Dataset sampler with `step_skip=10` |
| 7 | L2 predictive WM `f_H` | `Ĥ_{τ+1}=H_τ+Conv([H_τ,enc(z)])` | `ConvPredictor` `l2_d4rl_e_p` | `[B,18,43,43]`, `[B,8]` | `[B,18,43,43]` | PredictionObs+Proprio | MLP `f_H_phi(h, g_xy)` 31→29 | **MISSING** | `Level2WorldModel` |
| 8 | Latent `z` / action abstraction | high-level action | `sequence_predictor.py` posterior `32-32`; `misc.py::PosteriorContinuous` | `[B,20]` | `[B,8]` | implicit via prediction | none (`g_τ` is xy) | **MISSING** | `LatentActionPosterior` |
| 9 | L1 planner | online primitive MPC | `mppi_planner.py`, `mpc.py` | `h_t`, target repr | `a ∈ R^2` | none (planning) | BC MLP worker | **MISSING** | `Level1MPPI` |
| 10 | L2 planner | online latent MPC | same, `latent_actions=True` | `H_t`, goal repr | `z ∈ R^8` | none | candidate/`Q_H` manager | **MISSING** | `Level2MPPI` |
| 11 | Hierarchical MPPI | L2 plan → first predicted state → L1 plan → env | `two_lvl_planner.py`, `hmpc.py` | current image | env action `(2,)` | none | SoftReach over dataset subgoals | **MISSING** | `HierarchicalMPPI` |
| 12 | Goal representation | encoded goal **image** | `mpc.py::_encode_targets` + `NormEvalWrapper.get_target_obs` | render at xy* | flattened `o*` | none | raw `desired_goal` `(2,)` | **MISSING** | encode goal the same way as observations |
| 13 | Training objectives | L1 VICReg+IDM+proprio; L2 pred obs+proprio | `objectives/{vicreg,idm,prediction}.py` | JEPA ForwardResult | scalar | as named | BC / 1-step MSE / K-step MSE | **MISSING** | `hwm_faithful/losses` |
| 14 | Normalization | per-channel image, action, loc, vel; optional L2 z bounds at eval | `pldm_envs/utils/normalizer.py` | raw tensors | z-scored | stats hardset for maze2d | `StateNormalizer` on 29D | **PARTIAL** (idea exists, stats/domain differ) | Diverse Maze normalizer, do not reuse AntMaze stats |
| 15 | Offline dataset | 25 maps, 98×98 npy + `data.p` episodes | `D4RLDataset`, HF `kevinghst/maze2d-large-diverse-25maps` | — | windows of images/actions | — | Minari AntMaze umaze | **MISSING** | `diverse_maze.py` loader around existing `data.p` / `images.npy`; do not regenerate |
| 16 | Evaluation pipeline | probe + hierarchical MPC on medium/hard trials | `evaluation/evaluator.py`, `hmpc.py` | 40 envs/split | success rate | — | AntMaze env rollouts, 50 seeds | **MISSING** | `eval/diverse_maze_eval.py` |
| 17 | Success metric | `||xy-xy*||<0.5` | `ant_draw.py::CustomMazeEnv._is_goal_reached` | xy | 0/1 | — | AntMaze 0.5 m (same threshold, different env) | **PARTIAL** | Keep 0.5 on Diverse Maze xy |
| 18 | Checkpointing | `{model,optim,epoch,step}` + ensemble blobs; `load_l1_only` strips L2/posterior | `train.py::{save_model,maybe_load_model}` | — | `.ckpt` | — | separate `.pt` per module | **PARTIAL** | New ckpt format for faithful models; do not clobber `checkpoints/f_h_explicit.pt` |
| 19 | Horizon / depth | L2 `T2` shrinks as time passes; min 3; L1 hierarchical horizon fixed at skip 10; then 15-step L1 finish | `mpc.py` `plan_size` math; `final_trans_steps=15` | — | — | — | fixed `K=10` manager interval | **MISSING** | replicate receding L2 horizon + final L1 |
| 20 | L2 plan → L1 plan → env action | `pred_obs[1]` is L1 target; execute `a_0`; action_repeat 4; replan every 4 | `two_lvl_planner.py`, `mpc.py` loop | — | `a ∈ R^2` | — | manager outputs `g_xy`, worker outputs 8D torque every step | **MISSING** | exact control stack for Diverse Maze first |

### Why the missing pieces matter scientifically

- **Learned representation.** Original HWM plans in a VICReg-regularized visual latent. Raw 29D Ant coordinates are a different scientific object. Claiming “our HWM vs original HWM” without this is not comparable.
- **L1 predictive latent model.** `f_L` is what L1 MPPI uses as the short-horizon world model. Our MLP `f_L` cannot substitute because it lives in a different observation space and is not used for MPC.
- **L2 hierarchy + `z`.** This *is* the “explicit high-level world model.” Our `f_H_phi(s, g_xy)` predicts K-step state given a coordinate subgoal; original `f_H` predicts L1 features given a latent macro-action inferred from 10 primitive actions.
- **Hierarchical MPPI.** Original `pi_H`/`pi_L` are planners. Our `pi_H` is retrieval+`Q_H` and `pi_L` is BC. That is a different control algorithm, not an implementation detail.
- **Temporal abstraction.** Skip-10 plus `z` chunks is how L2 becomes a long-horizon model. Our `K=10` is only the worker execution interval.
- **Visual Diverse Maze task.** We already have an original-code number on this exact benchmark. AntMaze 29D cannot falsify a faithful HWM until this number is matched.

---

## 9. MUST MATCH

Scientifically required before we may say our implementation is comparable to original HWM on Diverse Maze:

1. **Learned visual encoder** (not `E(s)=s`), operating on 98×98 RGB + proprio vel.
2. **Two-level predictive hierarchy**: L1 predictor on primitive actions; L2 predictor on latent `z`.
3. **L2 identity-on-L1** (no second visual encoder in the ICML L2 config).
4. **Temporal abstraction** `step_skip = 10` with chunked 10-step action posteriors.
5. **Explicit L2 world model** `H_{τ+1} = f_H(H_τ, z_τ)` with residual conv (or a documented equivalent that predicts the same L1 features).
6. **Deterministic 8D `z`** inferred from actions at train time; sampled by MPPI at plan time.
7. **Hierarchical planning**: L2 MPPI → first predicted L2 state as L1 target → L1 MPPI → primitive action.
8. **Same task data**: `maze2d_large_diverse_25maps` train, `maze2d_large_diverse_probe` eval, 25 maps, image obs.
9. **Same success metric**: Euclidean xy error `< 0.5`.
10. **Same eval protocol family**: medium/hard start-target files, `n_envs = 40`, `action_repeat = 4`, `replan_every = 4`, binary goal reward.
11. **Two-stage training**: train/freeze L1, then train L2 (the released recipe).
12. **Planning in representation space** with goal **image** encoding, not raw xy cost (unless an ablation shows xy cost matches the original numbers, which would itself be a result).

---

## 10. SHOULD MATCH

Likely to affect numbers; small clean deviations are acceptable if documented:

1. Exact conv widths / `d4rl_a`, `d4rl_b_p`, `l2_d4rl_e_p` layer tables.
2. VICReg coefficients and IDM on L1.
3. Residual + GroupNorm + ReLU details; `predictor_ln` on L2.
4. MPPI hyperparameters: `num_samples`, `noise_sigma`, `lambda_`, medium/hard overrides.
5. `final_trans_steps = 15` flat L1 finish.
6. Hard-set normalization statistics vs recomputed stats.
7. Location prober architecture (needed for original plots; not for success).
8. `min_plan_length = 3`, receding `T2`, `sum_all_diffs`.
9. Action bounding (`min_step`/`max_step`, L2 latent percentile bounds at eval).
10. Batch size 128, Adam, `base_lr ≈ 0.0176`, 3 L1 epochs / 5 L2 epochs.
11. `compile_model` / `torch.compile` (original L2 yaml leaves default True in `TrainConfig` except L1 yaml sets False).

---

## 11. Implementation details we can safely change

Not scientifically load-bearing:

1. File layout, class names, OmegaConf vs dataclasses, wandb vs local logs.
2. Dead YAML fields: `backbone_width_factor`, L1 `input_dim: 4`, L2 VICReg block that is not in `objectives`.
3. `torch.compile`, CUDA hardcoded `.cuda()` vs a `device` argument.
4. Plotting, heatmaps, wandb videos.
5. Exact MPPI third-party code structure, as long as the algorithm (sampling, importance weights, receding horizon) is the same.
6. Ensemble predictor path (`ensemble_size = 1` in this config).
7. RSSM / Transformer / Wall-dataset `encode_actions` (sum of primitives) — unused here.
8. Checkpoint filename scheme, as long as L1 can be loaded frozen into L2.
9. DataLoader worker counts, prefetch, `num_workers`.
10. Whether we wrap existing `HierarchicalWorldModel` with an alias `HWMRawState` or leave the old class name and document the alias.

---

## 12. Proposed clean module structure

Do **not** copy `pldm/` files. Do **not** modify `src/hwm_director/`. Add a sibling package.

```
src/hwm_faithful/
  __init__.py                 # exports HWMFaithful; documents HWMRawState = existing class
  config.py                   # Diverse Maze / two-level hyperparams (our dataclasses)

  models/
    visual_encoder.py         # MeNet6-equivalent, d4rl_a topology
    proprio.py                # Expander2D late fusion
    conv_predictor.py         # residual conv predictor used by both levels
    latent.py                 # action posterior (20→8) and dummy/learned prior
    level1.py                 # encoder + f_L (primitive-action predictor)
    level2.py                 # identity-on-L1 backbone + f_H (z predictor)
    hjepa.py                  # thin wrapper: encode L1, skip, infer z, predict L2

  losses/
    prediction.py             # MSE on obs / proprio components
    vicreg.py                 # L1 visual VICReg
    idm.py                    # L1 inverse dynamics

  data/
    diverse_maze.py           # windows, skip=10, chunked actions, image mmap
    normalize.py              # image/action/vel/location stats

  planning/
    mppi.py                   # single-level MPPI over a predictor
    hierarchical.py           # L2 then L1, pred_obs[1] as subgoal
    cost.py                   # repr-target MSE on obs_component

  eval/
    diverse_maze_eval.py      # 40-env medium/hard, 0.5 success
    env_wrappers.py           # action_repeat, image obs, binary goal

scripts/
  train_hwm_faithful_l1.py
  train_hwm_faithful_l2.py
  eval_hwm_faithful_diverse_maze.py
```

Existing baseline stays:

```
src/hwm_director/             # HWMRawState / Director — untouched
scripts/train_hwm.py          # baseline HWM
scripts/train_director.py
checkpoints/f_h_explicit.pt   # baseline, gitignored
```

Optional later alias (no behavior change):

```python
# src/hwm_director/models/hwm.py
HierarchicalWorldModel  # keep
HWMRawState = HierarchicalWorldModel  # add when implementing, not before review
```

This structure follows the original *modules* (encoder, two JEPAs, posterior, MPPI, HMPC) without the original's Wall/RSSM/ensemble/transformer surface area.

---

## 13. Implementation milestones

Order is driven by code: L1 is a frozen encoder for L2, and hierarchical planning needs both predictors plus MPPI. Diverse Maze eval is the gate before any AntMaze port.

| ID | Milestone | Exit criterion |
| --- | --- | --- |
| **M0** | Dataset + observation pipeline parity (loader only; **no regeneration**) | Can iterate `data.p` + `images.npy`, produce L1 windows `[15,3,98,98]` and L2 windows `[7,3,98,98]` + chunked `[6,10,2]` actions; normalizer stats match hardset |
| **M1** | L1 visual encoder + proprio fusion | **IMPLEMENTED** (2026-09-09). Synthetic 98×98 forward: visual `[B,16,43,43]`, fused `[B,18,43,43]`. See Section 17. |
| **M2** | L1 residual conv world model | **IMPLEMENTED** (2026-09-09). Recursive 14-step rollout `[15,B,18,43,43]`; residual on all 18 fused channels; action expand to 2 spatial channels. See Section 18. |
| **M3** | L1 training + representation checks | VICRegObs+IDM+PredProprio on Diverse Maze; location probe error vs frozen random encoder; no AntMaze yet |
| **M4** | L1 MPPI | Flat L1 planning runs on a few envs without crash; actions `(2,)`; optional comparison to original L1-only yaml (not the 82.5/90 target) |
| **M5** | L2 skip + identity backbone + conv `f_H` | Skip-10 batch → L1 encode_only → L2 residual predict |
| **M6** | Latent `z` posterior | Train-time `z = posterior(actions_chunk)`; shape `[B,8]`; L2 conditioned on `z` not on `a` |
| **M7** | Hierarchical MPPI | `pred_obs[1]` as L1 target; replan_every=4; action_repeat=4; final 15-step L1 |
| **M8** | End-to-end Diverse Maze eval | medium + hard, 40 envs, success `< 0.5` m |
| **M9** | Compare to original-code reproduction | Target: medium **82.5%**, hard **90.0%**. Report gaps; do not tune on AntMaze |
| **M10** | Only after M9, port faithful pieces into Wu `M_hier` on AntMaze | New code paths; baseline 32%/22% numbers remain |

Suggested first implementation after this review: **M1 with a fake M0** (synthetic tensors + dataset *interface*), then real M0 when we are allowed to wire the existing HF files. Do not start M5–M10 first.

---

## 14. Which milestone to implement first

**First: M1 (encoder + shape tests) on synthetic 98×98 tensors**, with the M0 dataset API sketched but not bound to a multi-gigabyte `images.npy` until you approve data use.

Reason:

- Every later module's shapes are determined by the encoder (`16×43×43`).
- It needs no dataset download/regeneration.
- It is CPU-feasible and immediately falsifiable.
- It does not touch `hwm_director`.

Immediately after M1: M0 against the **already downloaded** original-repro files (read-only), still no training.

---

## 15. GPU / compute by milestone

No hour estimates. Relative cost only. PACE = Georgia Tech cluster GPUs.

| Milestone | Device | Likely GPU memory | Local GPU enough? | PACE? | Relative compute |
| --- | --- | --- | --- | --- | --- |
| M0 loader | CPU | — | yes | no | tiny (I/O bound; images.npy is multi-GB) |
| M1 encoder shapes | CPU or tiny GPU | ≪ 2 GB | yes | no | tiny |
| M2 L1 predictor shapes | CPU or tiny GPU | ≪ 4 GB | yes | no | tiny |
| M3 L1 training | GPU | ~8–16 GB at batch 128, T=15, 98×98 conv | maybe a single 16–24 GB GPU | **yes if local is small** | **medium** |
| M4 L1 MPPI | GPU | grows with `num_samples × horizon × 18×43×43`; 500×128 is heavy | unlikely for full 40-env | **yes** | **large** |
| M5–M6 L2 train (frozen L1 encode) | GPU | L1 encode of 7-frame windows + L2 conv; batch 128 | maybe | recommended | **medium** |
| M7 hierarchical MPPI | GPU | L2 2000×35 plus L1 500×10 per env; original uses `n_envs_batch_size=20` to avoid OOM | no | **required** | **large** |
| M8–M9 40-env medium+hard | GPU | same as M7 × many replans | no | **required** | **large** |
| M10 AntMaze port | TBD | TBD | TBD | TBD | after M9 only |

Original `Trainer` calls `self.model.cuda()` with no CPU fallback. Faithful training/eval should be treated as GPU-only.

---

## 16. Ambiguities / items that need more investigation

1. **`z_reg_coeff` is computed but not applied** in `mppi_torch.py::_compute_rollout_costs`. Is the paper's L2 latent regularizer actually on in the released code? Need a decision: match dead code (no reg) or match the YAML intention (add the term).
2. **`backbone_width_factor: 2` is unused.** Matching YAML vs matching executed channels (`d4rl_a` as written) — we should match executed channels.
3. **L2 VICReg is configured but not in `objectives`.** Confirm with authors/paper whether L2 VICReg was used in the reported ICML run. Released L2 yaml does not apply it.
4. **`n_steps: 500` vs binding `max_plan_length_l2 * skip`.** Medium uses 350+15 wrapper steps, not 500. Report should cite the executed horizon.
5. **TwoLvlPlanner passes L1 `BackboneOutput` into L2 MPPI with `repr_input=True`**, skipping L2 `IdentityEncoder`. Equivalent only because L2 is identity-on-L1. A clean reimplementation should call L2 encode explicitly.
6. **L1 hierarchical cost vs 16 vs 18 channels.** Cost uses `obs_component` (16 ch). `pred_encoder = L2.backbone` with `using_proprio=True` then slices 16+2 from a 16-channel tensor (empty proprio slice). It still matches `pred_obs[1]` (16 ch). We should implement the *intended* comparison (L1 visual map vs L2-predicted visual map) and note this slice quirk.
7. **Training `images.npy` for 25maps.** Original-repro currently has `data.p` and an `images/` directory; probe split has `images.npy` (2.8G). Confirm which image store the ICML run used before M3. Do not regenerate.
8. **Checkpoint vs YAML `load_checkpoint_path`.** Reproduction used `load_from_l1248-seed248_epoch=5_...ckpt`; L2 yaml comments a different path (`3-9-1-seed248_...`). Pin one.
9. **KL / stochastic `z`.** Code supports both; this config uses neither. Paper language may still say “latent.” Our writeups should say “deterministic action-inferred `z`.”
10. **Prober vs planner.** Official `eval_only` still trains location probers. Success does not use them. Decide whether M8 trains probers or only logs xy from the env.
11. **`error_threshold: 1.0` and `final_trans_norm_cutoff: 4`** in hierarchical config are not obviously used in `_perform_h_mpc`. Dead?
12. **Director sharing.** Original HWM has no neural `pi_L`. M10 must choose: (a) keep MPPI as `pi_L` on AntMaze pixels/state, or (b) replace L1 MPPI with the BC worker and only port `E`/`f_L`/`f_H`. That is a later scientific decision, not a coding one for M1–M9.

---

## Preservation constraints (repeated)

- Do not modify original HWM.
- Do not delete or overwrite `src/hwm_director/` simplified Director/HWM.
- Do not start training, PACE jobs, or image regeneration in this phase.
- Do not port into 29D AntMaze until Diverse Maze parity (M9).
- Baseline AntMaze numbers (Director ~32%, HWM ~22%) remain the raw-state baseline.

---

## 17. M1 implementation (Level-1 encoder)

**Status: IMPLEMENTED.** M2 is implemented separately (Section 18). M3+ remain not started.

### Files

```
src/hwm_faithful/shapes.py
src/hwm_faithful/__init__.py
src/hwm_faithful/models/__init__.py
src/hwm_faithful/models/visual_encoder.py   # VisualEncoder
src/hwm_faithful/models/proprio.py          # ProprioExpander
src/hwm_faithful/models/level1_encoder.py   # Level1Encoder
tests/test_hwm_faithful_level1_encoder.py
scripts/smoke_hwm_faithful_level1.py
```

`src/hwm_director/` was not modified.

### Original mapping

| Ours | Original |
| --- | --- |
| `VisualEncoder` | `MeNet6.layers` via `build_conv(ENCODER_LAYERS_CONFIG["d4rl_a"], (3,))` |
| `ProprioExpander` | `Expander2D` / late `id_expand` in `MeNet6._build_proprio_encoder` |
| `Level1Encoder` | `MeNet6.forward` with `early_proprio` ignored, `late_proprio.fuse=true` |
| `output.visual` | `BackboneOutput.obs_component` |
| `output.proprio_map` | `BackboneOutput.proprio_component` |
| `output.fused` | `BackboneOutput.encodings` = `cat([obs, late_proprio], dim=1)` |

### Layer sequence and shapes (`B` arbitrary)

| Step | Op | Output |
| --- | --- | --- |
| input | RGB | `[B, 3, 98, 98]` |
| block0 | Conv 3→16, k=5, s=1, p=0 + GN(4) + ReLU | `[B, 16, 94, 94]` |
| block1 | Conv 16→32, k=5, s=2, p=0 + GN(8) + ReLU | `[B, 32, 45, 45]` |
| block2 | Conv 32→32, k=3, s=1, p=0 + GN(8) + ReLU | `[B, 32, 43, 43]` |
| block3 | Conv 32→32, k=3, s=1, p=1 + GN(8) + ReLU | `[B, 32, 43, 43]` |
| block4 | Conv 32→16, k=1, s=1, p=0 (no GN, no ReLU) | `[B, 16, 43, 43]` |
| proprio | spatial broadcast of `[B, 2]` | `[B, 2, 43, 43]` |
| fused | `cat(visual, proprio_map, dim=1)` | `[B, 18, 43, 43]` |

First 16 fused channels are visual; last 2 are proprio (same as original).

### Parameter counts

| Module | Trainable params |
| --- | --- |
| `VisualEncoder` | 33,296 |
| `ProprioExpander` | 0 |
| `Level1Encoder` | 33,296 |

This matches the original executed topology: `d4rl_a` first-layer in-channels overridden from the table's 6 to RGB 3; GroupNorm affine on the first four convs; last 1×1 conv has bias and no norm. `backbone_width_factor: 2` is unused in original Python and unused here.

Live original check (read-only `hwm-original-repro`, `build_backbone(menet6, d4rl_a, proprio=2, 98×98)`): **33,296** trainable params, `output_dim=(18, 43, 43)`, forward `encodings [2,18,43,43]`. Same as ours.

### Implementation differences (intentional)

- Clean modules instead of copying `MeNet6` / `build_conv` / the full `ENCODER_LAYERS_CONFIG` table.
- No `.cuda()`; device follows input tensors.
- `ProprioExpander` uses `expand` + `contiguous` rather than `repeat`; values are identical (each scalar channel is constant over H×W).
- Typed public API (`Level1EncoderOutput`) instead of `BackboneOutput`.
- Explicit shape checks with `ValueError` (original asserted or failed inside conv).
- No early-proprio, location, or local-patch branches (those are off in the Diverse Maze L1 config).

### Tests

```
PYTHONPATH=src python -m pytest tests/test_hwm_faithful_level1_encoder.py -q
PYTHONPATH=src python scripts/smoke_hwm_faithful_level1.py
```

Results (CPU, 2026-09-09): `10 passed in 430.36s`. Smoke: fused `[4, 18, 43, 43]`, `visual=33296 proprio=0 total=33296`, `scalar loss 0.246135`, visual grads ok.

### Not in M1

Predictor / `f_L`, losses, L2, `z`, MPPI, datasets, training.

---

## 18. M2 implementation (Level-1 residual conv predictor)

**Status: IMPLEMENTED.** M3+ remain not started. Architecture and rollout semantics only; no training objectives.

### Files

```
src/hwm_faithful/models/action_encoder.py     # PrimitiveActionEncoder
src/hwm_faithful/models/conv_predictor.py     # Level1Predictor
src/hwm_faithful/models/level1_world_model.py # Level1WorldModel (E + f_L)
tests/test_hwm_faithful_level1_predictor.py
scripts/smoke_hwm_faithful_level1_predictor.py
src/hwm_faithful/shapes.py                    # ACTION_DIM, PREDICTOR_IN_CHANNELS
```

`src/hwm_director/` was not modified. Original HWM was not modified.

### Original mapping

| Ours | Original |
| --- | --- |
| `PrimitiveActionEncoder` | `ConvPredictor.action_encoder` = `Expander2D` when `action_encoder_arch='id'` |
| `Level1Predictor` | `ConvPredictor` (`predictor_arch=conv2`, subclass `d4rl_b_p`) |
| `Level1Predictor.compute_delta` | `self.layers(cat([h, a_map], dim=1))` |
| `fused = h + delta` | `ConvPredictor.forward`: `x = x + current_state` if `residual` |
| `obs_component` / `proprio_component` | `SequencePredictor._separate_obs_proprio_from_fused_repr` (last 2 channels = proprio) |
| `Level1Predictor.rollout` | `SequencePredictor.forward_multiple` |
| `Level1WorldModel` | `JEPA` encoder + predictor, L1 only (`z_dim=0`) |
| `Level1PredictorOutput` | `SingleStepPredictorOutput` |
| `Level1RolloutOutput` | `PredictorOutput` (predictions / obs / proprio) |

Factory path: `build_predictor` → `build_single_predictor` → `arch=="conv2"` → `ConvPredictor`. YAML `rnn_converter_arch`, `rnn_layers`, `rnn_state_dim` are unused for this arch. `predictor_ln` defaults false; `final_ln` is `Identity` and is **not applied** in `ConvPredictor.forward`.

### Action-conditioning path

```
a_t  [B, 2]
  -> Expander2D / PrimitiveActionEncoder (no MLP, 0 params)
  -> action_map [B, 2, 43, 43]
  -> cat([h_t, action_map], dim=1)   # h first, action last
  -> [B, 20, 43, 43]
```

`z_dim=0`: no latent, no prior/posterior. Primitive action only.

### Predictor layer sequence (`group_factor=8`, not the encoder's 4)

| Step | Op | Output |
| --- | --- | --- |
| h | fused L1 state | `[B, 18, 43, 43]` |
| a | primitive action | `[B, 2]` |
| expand | spatial broadcast | `[B, 2, 43, 43]` |
| concat | channel cat | `[B, 20, 43, 43]` |
| block0 | Conv 20→32, k=3, s=1, p=1 + GN(4) + ReLU | `[B, 32, 43, 43]` |
| block1 | Conv 32→32, k=3, s=1, p=1 + GN(4) + ReLU | `[B, 32, 43, 43]` |
| block2 | Conv 32→18, k=3, s=1, p=1 (no GN, no ReLU) | `[B, 18, 43, 43]` |
| residual | `h_next = h + delta` | `[B, 18, 43, 43]` |
| split | first 16 visual, last 2 proprio | `[B, 16, 43, 43]` + `[B, 2, 43, 43]` |

Table `d4rl_b_p` is `[(20,32,3,1,1), (32,32,3,1,1), (32,18,3,1,1)]`. `build_conv` still overrides first-layer in-channels to the actual concat width (20). Last conv has no GroupNorm/ReLU (`last_layer_act_norm=False`).

### Residual semantics (critical)

Original **`ConvPredictor`** (the executed L1 class) adds residual to the **full fused 18-channel state**:

```
delta = conv(cat(h, a_map))
h_next = h + delta
```

This includes proprio channels. Do not confuse with `ConvLocalPredictor`, which comments "only apply residual to obs" — that class is not used by Diverse Maze L1 (`predictor_arch=conv2`). No LayerNorm after the add.

### Rollout convention

Original `forward_multiple`:

- Input `state_encs[0]` = `h0`, `actions` length `T` (JEPA training: `T = n_steps - 1 = 14`)
- Loop uses **predicted** `current_state`, not ground-truth `state_encs[t]`
- Output `predictions` shape **`[T+1, B, 18, 43, 43]`**
- Index 0 is `h0`; index `t+1` is `f(h_t, a_t)`

Live original check: `T=3` → `predictions (4, B, 18, 43, 43)`, `rollout[0] == h`, `out == h + layers(cat(h, a_map))`.

### Parameter counts

| Module | Ours | Original |
| --- | --- | --- |
| action expander | 0 | 0 (`Expander2D`) |
| conv predictor | **20,370** | **20,370** (`ConvPredictor.layers`) |
| unused `final_ln` | omitted | `Identity` (0 params) |

Live original instantiation (`build_predictor(conv2, d4rl_b_p, residual=true, action_encoder_arch=id, z_dim=0)`): 20,370 trainable params. Same as ours.

Breakdown: Conv(20,32,3)+bias 5,792; GN(32) 64; Conv(32,32,3)+bias 9,248; GN(32) 64; Conv(32,18,3)+bias 5,202. Total 20,370.

World-model total = encoder 33,296 + predictor 20,370 = 53,666.

### Implementation differences (intentional)

- Clean modules instead of copying `ConvPredictor` / `SequencePredictor` / the full `ConvPredictorConfig` table.
- No unused `final_ln` Identity, prior/posterior, ensemble, RNN, or `z` heads (`z_dim=0`).
- `expand` + `contiguous` rather than `repeat` (values identical).
- Typed outputs; explicit `ValueError` on bad shapes.
- No `.cuda()`; device follows inputs.

### Tests

```
PYTHONPATH=src python -m pytest tests/test_hwm_faithful_level1_predictor.py -q
PYTHONPATH=src python scripts/smoke_hwm_faithful_level1_predictor.py
```

Results (CPU, 2026-09-09): `16 passed in 2345.10s`. Live original `ConvPredictor`: 20,370 params; one-step `[2,18,43,43]`; T=3 rollout `[4,2,18,43,43]`; residual `h + conv(cat(h,a_map))` holds.

Smoke: one-step `[4,18,43,43]`, 14-step rollout `[15,4,18,43,43]` with `rollout[0]=h0`, `predictor=20370 action=0 encoder=33296 world_model=53666`, `scalar loss 26.421385`, predictor grads ok, encode→predict `[4,18,43,43]`.

### Not in M2

VICReg / IDM / proprio losses, optimizer, training loop, dataset loader, Level 2, latent `z`, MPPI, hierarchical planning, AntMaze.

