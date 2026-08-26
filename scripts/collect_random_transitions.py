#!/usr/bin/env python3
"""Collect random AntMaze transitions ``(s_t, a_t, s_{t+1})``."""

from __future__ import annotations

import argparse

import numpy as np

from hwm_director.data.state import extract_state_and_goal
from hwm_director.data.statistics import compute_state_statistics, summarize_transitions
from hwm_director.data.transitions import Transition
from hwm_director.envs.antmaze import DEFAULT_ENV_ID, make_antmaze
from hwm_director.models.encoder import IdentityEncoder

SEED = 0
N_TRANSITIONS = 5000


def collect_random_transitions(
    n_transitions: int = N_TRANSITIONS, seed: int = SEED
) -> list[Transition]:
    """Roll out random actions and return ``n_transitions`` tuples in memory."""
    env = make_antmaze(DEFAULT_ENV_ID)
    encoder = IdentityEncoder()
    transitions: list[Transition] = []
    try:
        observation, _info = env.reset(seed=seed)
        env.action_space.seed(seed)
        while len(transitions) < n_transitions:
            action = env.action_space.sample()
            next_observation, _reward, terminated, truncated, _info = env.step(
                action
            )
            state, goal = extract_state_and_goal(observation)
            next_state, _ = extract_state_and_goal(next_observation)
            transition = Transition(
                state=encoder.encode(state),
                action=action,
                next_state=encoder.encode(next_state),
                goal=goal,
            )
            transition.validate()
            transitions.append(transition)
            if terminated or truncated:
                observation, _info = env.reset()
            else:
                observation = next_observation
    finally:
        env.close()
    return transitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-transitions",
        type=int,
        default=N_TRANSITIONS,
        help="number of (s, a, s') tuples to collect (default: 5000)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--print-state-stats",
        action="store_true",
        help="print per-dimension state mean, std, min, and max",
    )
    args = parser.parse_args()

    transitions = collect_random_transitions(args.n_transitions, args.seed)
    print("=== collection summary ===")
    print(f"  env_id={DEFAULT_ENV_ID}")
    print(f"  n_transitions={len(transitions)}")
    print(summarize_transitions(transitions))

    if args.print_state_stats:
        states = np.stack([t.state for t in transitions], axis=0)
        stats = compute_state_statistics(states)
        print("=== per-dimension state statistics ===")
        for key in ("mean", "std", "min", "max"):
            values = np.asarray(stats[key])
            print(f"  {key}: shape={values.shape}")
            print(f"    {values}")

    print("=== closed environment ===")


if __name__ == "__main__":
    main()
