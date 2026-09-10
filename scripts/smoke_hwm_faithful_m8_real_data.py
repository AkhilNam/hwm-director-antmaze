#!/usr/bin/env python3
"""M8 real-data / eval smoke. No training. No 40-env eval.

Uses probe ``images.npy`` (mmap) when present. Optional env reset/step and
one-action hierarchical plan with K=4.
"""

from __future__ import annotations

import os
import traceback

import torch

from hwm_faithful.data.diverse_maze import load_probe_dataset, probe_data_available, add_time_batch
from hwm_faithful.data.env import (
    DiverseMazeEnvAdapter,
    encode_goal,
    load_starts_targets,
    original_env_import_error,
    original_env_importable,
    success_from_distance,
)
from hwm_faithful.data.normalizer import ACTION_MEAN, ACTION_STD
from hwm_faithful.level2.temporal_abstraction import build_level2_inputs
from hwm_faithful.level2.world_model import Level2WorldModel
from hwm_faithful.losses.level1_objective import Level1Objective
from hwm_faithful.losses.level2_objective import level2_prediction_losses
from hwm_faithful.models.level1_world_model import Level1WorldModel
from hwm_faithful.planning.config import Level1MPPIConfig, Level2MPPIConfig
from hwm_faithful.planning.hierarchical_planner import HierarchicalPlanner


def _print_losses(prefix: str, losses) -> None:
    print(prefix)
    for name in losses._fields:
        val = getattr(losses, name)
        if torch.is_tensor(val) and val.ndim == 0:
            print(f"  {name:24s} {val.item():.6f} finite={bool(torch.isfinite(val))}")


def main() -> None:
    torch.manual_seed(0)
    if not probe_data_available():
        print("SKIP probe dataset not found")
        return
    ds = load_probe_dataset()
    info = ds.describe()
    print("dataset")
    for k, v in info.items():
        print(f"  {k}: {v}")

    w1 = ds.get_level1_window(0)
    print("\nL1 window")
    print("  images ", tuple(w1.images.shape), w1.images.dtype)
    print("  proprio", tuple(w1.proprio.shape))
    print("  actions", tuple(w1.actions.shape))
    print("  xy     ", tuple(w1.locations.shape))

    l1 = Level1WorldModel()
    l1.eval()
    images, proprio = add_time_batch(w1.images, w1.proprio)
    images = images.repeat(1, 2, 1, 1, 1)
    proprio = proprio.repeat(1, 2, 1)
    actions = w1.actions.unsqueeze(1).repeat(1, 2, 1)
    with torch.no_grad():
        enc, rolled = l1.encode_and_rollout(images, proprio, actions)
        l1_losses = Level1Objective()(
            obs_gt=enc.visual,
            obs_pred=rolled.obs_component,
            proprio_gt=enc.proprio_map,
            proprio_pred=rolled.proprio_component,
            fused_gt=enc.fused,
            actions=actions,
        )
    print("L1 encode fused", tuple(enc.fused.shape), "finite", bool(torch.isfinite(enc.fused).all()))
    print("L1 rollout     ", tuple(rolled.fused.shape), "finite", bool(torch.isfinite(rolled.fused).all()))
    _print_losses("L1 losses", l1_losses)

    w2 = ds.get_level2_primitive_window(0)
    print("\nL2 primitive window")
    print("  images ", tuple(w2.images.shape))
    print("  actions", tuple(w2.actions.shape))
    images2, proprio2 = add_time_batch(w2.images, w2.proprio)
    with torch.no_grad():
        enc2 = l1.encode_sequence(images2, proprio2)
        l2_in = build_level2_inputs(enc2.fused, w2.actions.unsqueeze(1))
        l2 = Level2WorldModel()
        l2.eval()
        fwd = l2.forward_train(l2_in.states, l2_in.action_chunks)
        l2_losses = level2_prediction_losses(fwd.h_gt, fwd.h_pred)
    print("  H_gt   ", tuple(fwd.h_gt.shape))
    print("  chunks ", tuple(l2_in.action_chunks.shape))
    print("  z      ", tuple(fwd.z.shape))
    print("  H_pred ", tuple(fwd.h_pred.shape))
    _print_losses("L2 losses", l2_losses)

    trials = load_starts_targets(difficulty="medium")
    t0 = trials.trial(0)
    print("\nstart/target[0]")
    print("  start", t0["start"], "target", t0["target"])
    print("  block_dist", t0["block_dist"], "turns", t0["turns"])
    print("  success(start,target)", success_from_distance(t0["start"], t0["target"]))

    # Hierarchical one-action on real encoded observations (no env required).
    w_cur = ds.get_level1_window(0)
    w_goal = ds.get_level1_window(10)
    with torch.no_grad():
        cur = l1.encode(w_cur.images[0:1], w_cur.proprio[0:1])
        goal_v = encode_goal(l1.encoder, w_goal.images[0:1])
        planner = HierarchicalPlanner(
            l1.predictor,
            l2.predictor,
            l1_config=Level1MPPIConfig(num_samples=4),
            l2_config=Level2MPPIConfig(num_samples=4, max_plan_length=3),
            n_envs=1,
            action_mean=ACTION_MEAN,
            action_std=ACTION_STD,
        )
        plan = planner.plan(
            current_fused_state=cur.fused,
            final_goal_visual=goal_v,
            l2_horizon=3,
            l1_horizon=10,
        )
    a0 = plan.level1.action_sequence[0, 0]
    print("\nhierarchical one-action (real encoded frames, no env)")
    print("  l1_target", tuple(plan.l1_target.shape))
    print("  primitives", tuple(plan.level1.action_sequence.shape))
    print("  a0", a0.tolist(), "finite", bool(torch.isfinite(a0).all()))

    if os.environ.get("HWM_M8_SKIP_ENV"):
        print("\nSKIP env smoke (HWM_M8_SKIP_ENV)")
        return
    if not original_env_importable():
        print("\nSKIP env:", original_env_import_error())
        return
    try:
        adapter = DiverseMazeEnvAdapter.from_stored_trial(difficulty="medium", trial_index=0)
        obs = adapter.reset()
        print("\nenv reset")
        print("  image  ", tuple(obs.image.shape), obs.image.dtype)
        print("  proprio", tuple(obs.proprio.shape), float(obs.proprio[0]), float(obs.proprio[1]))
        print("  xy     ", obs.xy, "goal", obs.goal_xy)
        encoded = adapter.encode_current(l1.encoder, obs)
        goal_v = adapter.encode_goal(l1.encoder, obs)
        print("  fused  ", tuple(encoded.fused.shape), "goal visual", tuple(goal_v.shape))
        step = adapter.step_normalized(torch.zeros(2))
        print("  step0  xy", step.xy, "reward", step.reward, "finite image", bool(torch.isfinite(step.image).all()))

        planner = HierarchicalPlanner(
            l1.predictor,
            l2.predictor,
            l1_config=Level1MPPIConfig(num_samples=4),
            l2_config=Level2MPPIConfig(num_samples=4, max_plan_length=3),
            n_envs=1,
            action_mean=ACTION_MEAN,
            action_std=ACTION_STD,
        )
        plan = planner.plan(
            current_fused_state=encoded.fused,
            final_goal_visual=goal_v,
            l2_horizon=3,
            l1_horizon=10,
        )
        a0 = plan.level1.action_sequence[0, 0]
        print("\nhierarchical one-action")
        print("  l1_target", tuple(plan.l1_target.shape))
        print("  primitive", tuple(plan.level1.action_sequence.shape), "a0", a0.tolist())
        stepped = adapter.step_raw(a0)
        print("  after plan step xy", stepped.xy, "reward", stepped.reward)
        adapter.close()
    except Exception:
        print("\nENV SMOKE FAILED")
        traceback.print_exc()


if __name__ == "__main__":
    main()
