#!/usr/bin/env python3
"""Train one-step low-level dynamics ``f_L`` on random AntMaze transitions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hwm_director.models.dynamics_low import LowLevelDynamicsModel
from hwm_director.training.train_dynamics import train_low_level_dynamics

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from collect_random_transitions import collect_random_transitions  # noqa: E402

SEED = 0
N_TRANSITIONS = 5000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-transitions", type=int, default=N_TRANSITIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256],
        help="MLP hidden layer widths",
    )
    args = parser.parse_args()

    transitions = collect_random_transitions(args.n_transitions, args.seed)
    print(f"collected {len(transitions)} transitions")

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
    print(f"  train MSE: {train_mse:.6f}")
    print(f"  val MSE: {val_mse:.6f}")
    print(f"  no-change baseline val MSE: {no_change:.6f}")
    print(f"  val x/y MSE: {val_xy:.6f}")
    print(f"  learned val MSE {vs_baseline} no-change ({val_mse:.6f} vs {no_change:.6f})")


if __name__ == "__main__":
    main()
