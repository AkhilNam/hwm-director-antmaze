#!/usr/bin/env python3
"""Train one-step low-level dynamics ``f_L`` on Minari AntMaze transitions.

Default data: ``D4RL/antmaze/umaze-v1``. Random rollouts are not used here;
see ``scripts/collect_random_transitions.py`` only as a smoke-test utility.
"""

from __future__ import annotations

import argparse

from hwm_director.data.minari_antmaze import (
    DEFAULT_MINARI_DATASET_ID,
    load_minari_transitions,
)
from hwm_director.models.dynamics_low import LowLevelDynamicsModel
from hwm_director.training.checkpoints import save_dynamics_checkpoint
from hwm_director.training.train_dynamics import train_low_level_dynamics

SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_MINARI_DATASET_ID)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="optional cap on Minari episodes (debug)",
    )
    parser.add_argument(
        "--max-transitions",
        type=int,
        default=None,
        help="optional cap on converted transitions (debug)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256],
        help="MLP hidden layer widths",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print train/validation episode IDs",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="write f_L weights + normalizer after training",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="checkpoints/f_l.pt",
        help="path used with --save-checkpoint",
    )
    args = parser.parse_args()

    transitions = load_minari_transitions(
        args.dataset_id,
        max_episodes=args.max_episodes,
        max_transitions=args.max_transitions,
    )
    n_episodes = len({t.episode_id for t in transitions})
    print(
        f"loaded {len(transitions)} transitions across {n_episodes} episodes "
        f"from {args.dataset_id}"
    )
    if transitions:
        t0 = transitions[0]
        print(
            f"  state={t0.state.shape} action={t0.action.shape} "
            f"next_state={t0.next_state.shape} goal={t0.goal.shape}"
        )

    print(
        "starting f_L training (preprocess then "
        f"{args.epochs} epochs on {len(transitions)} transitions)",
        flush=True,
    )
    model = LowLevelDynamicsModel(hidden_dims=tuple(args.hidden_dims))
    metrics = train_low_level_dynamics(
        transitions,
        model=model,
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )

    train_mse = metrics["train_mse"]
    val_mse = metrics["val_mse"]
    no_change = metrics["no_change_val_mse"]
    val_xy = metrics["val_xy_mse"]
    vs_baseline = "beats" if val_mse < no_change else "does not beat"

    print("=== f_L report ===")
    print(f"  n_episodes: {metrics['n_episodes']}")
    print(f"  n_train_episodes: {metrics['n_train_episodes']}")
    print(f"  n_val_episodes: {metrics['n_val_episodes']}")
    print(f"  n_train_transitions: {metrics['n_train_transitions']}")
    print(f"  n_val_transitions: {metrics['n_val_transitions']}")
    if args.verbose:
        print(f"  train_episode_ids: {metrics['train_episode_ids']}")
        print(f"  val_episode_ids: {metrics['val_episode_ids']}")
    print(f"  train MSE: {train_mse:.6f}")
    print(f"  val MSE: {val_mse:.6f}")
    print(f"  no-change baseline val MSE: {no_change:.6f}")
    print(f"  val x/y MSE: {val_xy:.6f}")
    print(f"  learned val MSE {vs_baseline} no-change ({val_mse:.6f} vs {no_change:.6f})")

    if args.save_checkpoint:
        save_dynamics_checkpoint(
            args.checkpoint_path, metrics["model"], metrics["normalizer"]
        )
        print(f"saved f_L checkpoint to {args.checkpoint_path}")


if __name__ == "__main__":
    main()
