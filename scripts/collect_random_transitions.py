#!/usr/bin/env python3
"""Collect random AntMaze transitions ``(s_t, a_t, s_{t+1})``."""

from __future__ import annotations

from hwm_director.data.state import extract_state_and_goal
from hwm_director.data.statistics import summarize_transitions
from hwm_director.data.transitions import Transition
from hwm_director.envs.antmaze import DEFAULT_ENV_ID, make_antmaze
from hwm_director.models.encoder import IdentityEncoder

SEED = 0
N_TRANSITIONS = 100


def main() -> None:
    env = make_antmaze(DEFAULT_ENV_ID)
    encoder = IdentityEncoder()
    transitions: list[Transition] = []

    try:
        observation, info = env.reset(seed=SEED)
        env.action_space.seed(SEED)
        print(f"reset seed={SEED} info_success={info.get('success')}")

        while len(transitions) < N_TRANSITIONS:
            action = env.action_space.sample()
            next_observation, _reward, terminated, truncated, info = env.step(
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
                observation, info = env.reset()
            else:
                observation = next_observation

        print("=== collection summary ===")
        print(f"  env_id={DEFAULT_ENV_ID}")
        print(f"  n_transitions={len(transitions)}")
        print(summarize_transitions(transitions))
    finally:
        env.close()
        print("=== closed environment ===")


if __name__ == "__main__":
    main()
