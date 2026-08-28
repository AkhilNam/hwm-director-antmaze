#!/usr/bin/env python3
"""Train the shared low-level worker ``pi_L(s_t, g_tau) -> a_t``.

Default data: Minari ``D4RL/antmaze/umaze-v1``. Random rollouts are not used
here; see ``scripts/collect_random_transitions.py`` only as a smoke-test
utility.
"""

from __future__ import annotations

import argparse

from hwm_director.data.minari_antmaze import (
    DEFAULT_MINARI_DATASET_ID,
    load_minari_transitions,
)
from hwm_director.data.worker_dataset import DEFAULT_HORIZON_K
from hwm_director.models.worker import GoalConditionedWorker
from hwm_director.training.checkpoints import save_worker_checkpoint
from hwm_director.training.train_worker import (
    DEFAULT_MAX_SUBGOAL_DISTANCE,
    DEFAULT_MIN_SUBGOAL_DISTANCE,
    evaluate_worker_on_recorded_subgoals,
    train_goal_conditioned_worker,
)

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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="BC minibatch size (4096 is appropriate for the full 1e6-step "
        "Minari corpus; K=10 yields ~10 examples per step)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--horizon-k",
        type=int,
        default=DEFAULT_HORIZON_K,
        help="low-level horizon K (future offset and eval rollout cap)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256],
        help="MLP hidden layer widths",
    )
    parser.add_argument(
        "--n-eval-trials",
        type=int,
        default=20,
        help="subgoal-reaching trials after BC (0 to skip)",
    )
    parser.add_argument(
        "--min-subgoal-distance",
        type=float,
        default=DEFAULT_MIN_SUBGOAL_DISTANCE,
        help="meters; skip eval targets closer than this",
    )
    parser.add_argument(
        "--max-subgoal-distance",
        type=float,
        default=DEFAULT_MAX_SUBGOAL_DISTANCE,
        help="meters; skip eval targets farther than this",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.5,
        help="meters; eval success if final distance is below this",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print episode IDs and per-trial eval details",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="write pi_L weights + normalizer after training",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="checkpoints/pi_l.pt",
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
        f"from {args.dataset_id}",
        flush=True,
    )
    if transitions:
        t0 = transitions[0]
        print(
            f"  state={t0.state.shape} action={t0.action.shape} "
            f"next_state={t0.next_state.shape} goal={t0.goal.shape}",
            flush=True,
        )
        print(
            f"  next: episode split + worker BC dataset with K={args.horizon_k} "
            f"(about {args.horizon_k}x more examples than transitions)",
            flush=True,
        )

    model = GoalConditionedWorker(hidden_dims=tuple(args.hidden_dims))
    metrics = train_goal_conditioned_worker(
        transitions,
        model=model,
        horizon_k=args.horizon_k,
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        log=lambda message: print(message, flush=True),
    )

    train_mse = metrics["train_mse"]
    val_mse = metrics["val_mse"]
    zero_mse = metrics["zero_action_val_mse"]
    vs_baseline = "beats" if val_mse < zero_mse else "does not beat"

    print("=== pi_L BC report ===")
    print(f"  n_episodes: {metrics['n_episodes']}")
    print(f"  n_train_episodes: {metrics['n_train_episodes']}")
    print(f"  n_val_episodes: {metrics['n_val_episodes']}")
    print(f"  n_train_transitions: {metrics['n_train_transitions']}")
    print(f"  n_val_transitions: {metrics['n_val_transitions']}")
    print(f"  n_train_examples: {metrics['n_train_examples']}")
    print(f"  n_val_examples: {metrics['n_val_examples']}")
    if args.verbose:
        print(f"  train_episode_ids: {metrics['train_episode_ids']}")
        print(f"  val_episode_ids: {metrics['val_episode_ids']}")
    print(f"  train action MSE: {train_mse:.6f}")
    print(f"  val action MSE: {val_mse:.6f}")
    print(f"  zero-action val MSE: {zero_mse:.6f}")
    print(
        f"  learned val MSE {vs_baseline} zero-action "
        f"({val_mse:.6f} vs {zero_mse:.6f})"
    )

    if args.save_checkpoint:
        save_worker_checkpoint(
            args.checkpoint_path, metrics["model"], metrics["normalizer"]
        )
        print(f"saved pi_L checkpoint to {args.checkpoint_path}", flush=True)

    if args.n_eval_trials > 0:
        val_ids = set(metrics["val_episode_ids"])
        val_transitions = [t for t in transitions if t.episode_id in val_ids]
        eval_metrics = evaluate_worker_on_recorded_subgoals(
            val_transitions,
            metrics["model"],
            metrics["normalizer"],
            horizon_k=args.horizon_k,
            n_trials=args.n_eval_trials,
            success_threshold=args.success_threshold,
            min_distance=args.min_subgoal_distance,
            max_distance=args.max_subgoal_distance,
            seed=args.seed,
            verbose=args.verbose,
            dataset_id=args.dataset_id,
        )
        if eval_metrics.get("skipped"):
            print("=== closed-loop controller comparison ===")
            print("  SKIPPED: exact MuJoCo restore is unavailable.")
            print(f"  reason: {eval_metrics['skip_reason']}")
            print(f"  {eval_metrics['todo']}")
            return
        print("=== closed-loop controller comparison ===")
        print(f"  n_candidates: {eval_metrics['n_candidates']}")
        print(f"  n_trials: {eval_metrics['n_trials']}")
        for name in ("worker", "zero", "random"):
            block = eval_metrics[name]
            print(f"{name}:")
            print(f"  mean final distance: {block['mean_final_distance']:.6f}")
            print(f"  mean distance reduction: {block['mean_distance_reduction']:.6f}")
            print(f"  progress fraction: {block['progress_fraction']:.6f}")
            print(f"  success rate: {block['success_rate']:.6f}")
        worker_final = eval_metrics["worker"]["mean_final_distance"]
        zero_final = eval_metrics["zero"]["mean_final_distance"]
        random_final = eval_metrics["random"]["mean_final_distance"]
        print(
            "worker improvement over zero: "
            f"{zero_final - worker_final:.6f}"
        )
        print(
            "worker improvement over random: "
            f"{random_final - worker_final:.6f}"
        )


if __name__ == "__main__":
    main()
