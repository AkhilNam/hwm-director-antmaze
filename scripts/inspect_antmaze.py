#!/usr/bin/env python3
"""Print AntMaze_UMaze-v5 spaces and a short random-action rollout."""

from __future__ import annotations

import numpy as np

from hwm_director.envs.antmaze import DEFAULT_ENV_ID, make_antmaze

SEED = 0
N_RANDOM_ACTIONS = 10
SAMPLE_ELEMS = 8


def _array_summary(name: str, value) -> None:
    arr = np.asarray(value)
    flat = arr.reshape(-1)
    sample = flat[:SAMPLE_ELEMS]
    print(f"  {name}:")
    print(f"    type={type(value).__name__} shape={arr.shape} dtype={arr.dtype}")
    print(f"    sample={sample}")


def _goal_distance(observation: dict) -> float:
    achieved = np.asarray(observation["achieved_goal"], dtype=np.float64)
    desired = np.asarray(observation["desired_goal"], dtype=np.float64)
    return float(np.linalg.norm(achieved - desired))


def _print_observation(observation: dict, heading: str) -> None:
    print(heading)
    print(f"  keys={list(observation.keys())}")
    for key, value in observation.items():
        _array_summary(key, value)
    print("  achieved_goal (explicit):", observation["achieved_goal"])
    print("  desired_goal  (explicit):", observation["desired_goal"])
    dist = _goal_distance(observation)
    print(f"  Euclidean ||achieved_goal - desired_goal|| = {dist:.6f}")


def main() -> None:
    env = make_antmaze(DEFAULT_ENV_ID)
    try:
        print("=== environment metadata ===")
        spec = env.spec
        print(f"  env_id={spec.id if spec is not None else DEFAULT_ENV_ID}")
        print(f"  spec={spec}")
        print(f"  max_episode_steps={spec.max_episode_steps if spec is not None else None}")
        print(f"  unwrapped={type(env.unwrapped).__name__}")

        print("=== spaces ===")
        print(f"  observation_space={env.observation_space}")
        print(f"  action_space={env.action_space}")
        print(f"  action_space.shape={env.action_space.shape} dtype={env.action_space.dtype}")

        observation, info = env.reset(seed=SEED)
        env.action_space.seed(SEED)

        print("=== reset ===")
        print(f"  seed={SEED}")
        print(f"  info={info}")
        _print_observation(observation, "=== observation (after reset) ===")

        print(f"=== {N_RANDOM_ACTIONS} random actions ===")
        for step in range(N_RANDOM_ACTIONS):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            dist = _goal_distance(observation)
            success = info.get("success", "<missing>")
            print(f"--- step {step} ---")
            print(
                f"  action: type={type(action).__name__} "
                f"shape={np.asarray(action).shape} dtype={np.asarray(action).dtype}"
            )
            print(f"  reward: type={type(reward).__name__} value={reward}")
            print(f"  terminated: type={type(terminated).__name__} value={terminated}")
            print(f"  truncated: type={type(truncated).__name__} value={truncated}")
            print(f"  info['success']={success}")
            print(f"  Euclidean ||achieved_goal - desired_goal|| = {dist:.6f}")
            if terminated or truncated:
                print("  episode ended; resetting for remaining inspect steps")
                observation, info = env.reset()
                print(f"  reset info={info}")
    finally:
        env.close()
        print("=== closed environment ===")


if __name__ == "__main__":
    main()
