#!/usr/bin/env python3
"""Inspect Minari ``D4RL/antmaze/umaze-v1`` without training.

Prints dataset size, one-episode structure, and the recovered environment.
Does not fit ``f_L`` or ``pi_L``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

import numpy as np

from hwm_director.envs.antmaze import DEFAULT_DATASET_ID


def _shape_dtype(value) -> str:
    arr = np.asarray(value)
    return f"type={type(value).__name__} shape={arr.shape} dtype={arr.dtype}"


def _print_mapping(name: str, value, indent: str = "  ") -> None:
    print(f"{indent}{name}: {_shape_dtype(value)}")
    if isinstance(value, Mapping):
        for key, inner in value.items():
            _print_mapping(str(key), inner, indent + "  ")
        return
    if isinstance(value, np.ndarray) and value.dtype == object and value.size:
        _print_mapping(f"{name}[0]", value.reshape(-1)[0], indent + "  ")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
        first = value[0]
        if isinstance(first, Mapping) or isinstance(first, np.ndarray):
            _print_mapping(f"{name}[0]", first, indent + "  ")


def _first_episode(dataset):
    if hasattr(dataset, "iterate_episodes"):
        return next(iter(dataset.iterate_episodes()))
    return dataset[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="do not download if the dataset is missing locally",
    )
    args = parser.parse_args()

    import minari

    dataset = minari.load_dataset(args.dataset_id, download=not args.no_download)
    print("=== Minari dataset ===")
    print(f"  dataset_id={getattr(dataset, 'id', args.dataset_id)}")
    spec = getattr(dataset, "spec", None)
    print(f"  spec={spec}")
    n_episodes = getattr(dataset, "total_episodes", None)
    if n_episodes is None:
        n_episodes = len(dataset)
    n_transitions = getattr(dataset, "total_steps", None)
    print(f"  n_episodes={n_episodes}")
    print(f"  n_transitions/total_steps={n_transitions}")

    episode = _first_episode(dataset)
    print("=== one episode ===")
    print(f"  episode_type={type(episode).__name__}")
    print(f"  episode.id={getattr(episode, 'id', None)}")
    print(f"  observations: {_shape_dtype(episode.observations)}")
    _print_mapping("observations", episode.observations)
    print(f"  actions: {_shape_dtype(episode.actions)}")
    print(f"  rewards: {_shape_dtype(episode.rewards)}")
    print(f"  terminations: {_shape_dtype(getattr(episode, 'terminations', None))}")
    print(f"  truncations: {_shape_dtype(getattr(episode, 'truncations', None))}")
    infos = getattr(episode, "infos", None)
    print(f"  infos: {_shape_dtype(infos) if infos is not None else None}")
    if isinstance(infos, Mapping):
        print(f"  infos.keys={list(infos.keys())}")
        for key, value in infos.items():
            print(f"    infos[{key!r}]: {_shape_dtype(value)}")
    elif isinstance(infos, Sequence) and infos:
        first = infos[0]
        print(f"  infos[0] type={type(first).__name__}")
        if isinstance(first, Mapping):
            print(f"  infos[0].keys={list(first.keys())}")

    print("=== recovered environment ===")
    env = dataset.recover_environment()
    try:
        spec = env.spec
        print(f"  env_id={spec.id if spec is not None else None}")
        print(f"  spec={spec}")
        print(f"  max_episode_steps={spec.max_episode_steps if spec is not None else None}")
        print(f"  observation_space={env.observation_space}")
        print(f"  action_space={env.action_space}")
        print(f"  unwrapped={type(env.unwrapped).__name__}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
