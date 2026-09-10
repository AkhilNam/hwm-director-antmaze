"""CLI for faithful Level-1 training, smoke, diagnostics, and env check.

Paths come from flags / environment, never from model source files.

Examples:

    python -m hwm_faithful.training.cli --mode overfit \\
        --data-path .../maze2d_large_diverse_probe/data.p \\
        --images-path .../images.npy \\
        --output-dir checkpoints/hwm_faithful/l1_overfit

    python -m hwm_faithful.training.cli --mode train \\
        --data-path .../maze2d_large_diverse_25maps/data.p \\
        --images-path .../images.npy \\
        --output-dir checkpoints/hwm_faithful/l1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hwm_faithful.data.diverse_maze import DiverseMazePaths
from hwm_faithful.data.env import (
    DiverseMazeEnvAdapter,
    original_env_import_error,
    original_env_importable,
)
from hwm_faithful.data.normalizer import ACTION_MEAN, ACTION_STD
from hwm_faithful.models.level1_world_model import Level1WorldModel
from hwm_faithful.planning.config import Level1MPPIConfig
from hwm_faithful.planning.level1_planner import Level1MPPIPlanner
from hwm_faithful.training.checkpoint import (
    apply_checkpoint,
    load_checkpoint,
    pick_latest_checkpoint,
    write_json,
)
from hwm_faithful.training.config import (
    DEAD_YAML_FIELDS,
    Level1TrainConfig,
    overfit_config,
    resolve_device,
    smoke_config,
)
from hwm_faithful.training.dataset import (
    describe_images,
    locate_25maps_images,
    make_level1_loader,
)
from hwm_faithful.training.diagnostics import evaluate_loader_batch
from hwm_faithful.training.level1_trainer import Level1Trainer, summarize_overfit


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str), flush=True)


def _default_probe_paths() -> tuple[str | None, str | None]:
    p = DiverseMazePaths.probe()
    data = str(p.pickle_path) if p.pickle_path.is_file() else None
    images = str(p.images_path) if p.images_path is not None else None
    return data, images


def build_argparser() -> argparse.ArgumentParser:
    probe_data, probe_images = _default_probe_paths()
    p = argparse.ArgumentParser(description="Faithful HWM Level-1 trainer")
    p.add_argument(
        "--mode",
        required=True,
        choices=(
            "train",
            "overfit",
            "smoke",
            "env-smoke",
            "diagnose",
            "plan-smoke",
            "describe-images",
        ),
    )
    p.add_argument("--data-path", default=None)
    p.add_argument("--images-path", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", default=None, dest="resume_path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--require-25maps", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--checkpoint", default=None, help="for diagnose / plan-smoke")
    p.add_argument("--plan-k", type=int, default=32)
    p.add_argument("--plan-horizon", type=int, default=10)
    p.add_argument("--probe-env", action="store_true")
    return p


def _apply_cli(cfg: Level1TrainConfig, args: argparse.Namespace) -> Level1TrainConfig:
    probe_data, probe_images = _default_probe_paths()
    cfg.data_path = args.data_path or cfg.data_path or probe_data
    cfg.images_path = args.images_path or cfg.images_path or probe_images
    cfg.output_dir = args.output_dir or cfg.output_dir
    cfg.resume_path = args.resume_path or cfg.resume_path
    cfg.device = resolve_device(args.device)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.max_windows is not None:
        cfg.max_windows = args.max_windows
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.require_25maps:
        cfg.require_25maps = True
    if args.no_resume:
        cfg.resume_if_possible = False
        cfg.resume_path = None
    cfg.pin_memory = cfg.device == "cuda"
    return cfg


def cmd_describe_images(args: argparse.Namespace) -> int:
    path = args.images_path or locate_25maps_images()
    if path is None:
        found = locate_25maps_images()
        _print_json(
            {
                "found": False,
                "explicit": args.images_path,
                "locate_25maps_images": None if found is None else str(found),
                "blocker": (
                    "maze2d_large_diverse_25maps/images.npy was not found. "
                    "Do not regenerate. Full L1 training cannot start."
                ),
            }
        )
        return 2
    info = describe_images(path)
    _print_json(info)
    return 0 if info["matches_25map_expectation"] else 3


def cmd_env_smoke(_args: argparse.Namespace) -> int:
    t0 = time.time()
    if not original_env_importable():
        payload = {
            "ok": False,
            "importable": False,
            "error": original_env_import_error(),
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        }
        _print_json(payload)
        return 1
    adapter = DiverseMazeEnvAdapter.from_stored_trial(difficulty="medium", trial_index=0)
    obs0 = adapter.reset()
    action = np.array([0.1, -0.05], dtype=np.float32)
    obs1 = adapter.step_raw(action)
    ok = (
        obs0.image.shape == (3, 98, 98)
        and obs1.image.shape == (3, 98, 98)
        and obs0.proprio.shape[-1] == 2
        and obs1.proprio.shape[-1] == 2
        and math.isfinite(obs1.reward)
        and torch.isfinite(obs0.image).all()
        and torch.isfinite(obs1.image).all()
        and torch.isfinite(obs0.proprio).all()
        and torch.isfinite(obs1.proprio).all()
    )
    payload = {
        "ok": bool(ok),
        "importable": True,
        "reset_image_shape": list(obs0.image.shape),
        "reset_proprio_shape": list(obs0.proprio.shape),
        "step_image_shape": list(obs1.image.shape),
        "step_proprio_shape": list(obs1.proprio.shape),
        "reward": float(obs1.reward),
        "done": bool(obs1.done),
        "xy0": obs0.xy.tolist(),
        "xy1": obs1.xy.tolist(),
        "action": action.tolist(),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "elapsed_s": time.time() - t0,
        "episode_run": False,
    }
    _print_json(payload)
    if args_output := getattr(_args, "output_dir", None):
        write_json(Path(args_output) / "env_smoke.json", payload)
    return 0 if ok else 1


def cmd_train(cfg: Level1TrainConfig, *, label: str) -> int:
    if cfg.output_dir is None:
        raise SystemExit("--output-dir is required")
    if cfg.require_25maps and locate_25maps_images(cfg.images_path) is None:
        print(
            "BLOCKER: 25-map images.npy not found. Refusing full L1 training.",
            file=sys.stderr,
        )
        return 2
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", cfg.to_dict())
    write_json(out / "dead_yaml_fields.json", DEAD_YAML_FIELDS)
    trainer = Level1Trainer(cfg)
    history = trainer.train()
    summary = {
        "label": label,
        "n_logged": len(history),
        "last": history[-1] if history else None,
        "output_dir": str(out),
        "device": cfg.device,
    }
    if label == "overfit" and history:
        summary["overfit"] = summarize_overfit(history)
    write_json(out / "train_summary.json", summary)
    _print_json(summary)
    return 0


def cmd_diagnose(args: argparse.Namespace, cfg: Level1TrainConfig) -> int:
    ckpt = args.checkpoint or (
        pick_latest_checkpoint(cfg.output_dir) if cfg.output_dir else None
    )
    if ckpt is None:
        raise SystemExit("diagnose: pass --checkpoint or --output-dir with latest.pt")
    model, _blob = _load_trained(ckpt, cfg.device)
    loader = make_level1_loader(cfg)
    batch = next(iter(loader))
    report = evaluate_loader_batch(model, batch, torch.device(cfg.device))
    if cfg.output_dir:
        write_json(Path(cfg.output_dir) / "diagnostics.json", report)
    _print_json(report)
    return 0


def _load_trained(checkpoint: str | Path, device: str) -> tuple[Level1WorldModel, Any]:
    from hwm_faithful.losses.level1_objective import Level1Objective

    blob = load_checkpoint(checkpoint, map_location=device)
    model = Level1WorldModel().to(device)
    obj = Level1Objective().to(device)
    apply_checkpoint(
        blob,
        encoder=model.encoder,
        predictor=model.predictor,
        idm=obj.idm,
        optimizer=None,
        scheduler=None,
    )
    model.eval()
    return model, blob


def cmd_plan_smoke(args: argparse.Namespace, cfg: Level1TrainConfig) -> int:
    ckpt = args.checkpoint or (
        pick_latest_checkpoint(cfg.output_dir) if cfg.output_dir else None
    )
    if ckpt is None:
        raise SystemExit("plan-smoke: pass --checkpoint or --output-dir")
    model, _ = _load_trained(ckpt, cfg.device)
    loader = make_level1_loader(cfg)
    batch = next(iter(loader))
    images = batch["images"].to(cfg.device).transpose(0, 1)
    proprio = batch["proprio"].to(cfg.device).transpose(0, 1)
    encoded = model.encode_sequence(images[:, :1], proprio[:, :1])
    h0 = encoded.fused[0]
    goal_visual = encoded.visual[-1]
    planner = Level1MPPIPlanner(
        model.predictor,
        Level1MPPIConfig(num_samples=int(args.plan_k)),
        action_mean=ACTION_MEAN.to(cfg.device),
        action_std=ACTION_STD.to(cfg.device),
    )
    result = planner.plan(
        h0, goal_visual, planning_horizon=int(args.plan_horizon)
    )
    init_c = float(result.diagnostics["initial_exec_cost"].reshape(-1)[0].item())
    opt_c = float(result.cost.reshape(-1)[0].item())
    actions = result.action_sequence[0].detach().cpu()
    payload: dict[str, Any] = {
        "initial_cost": init_c,
        "optimized_cost": opt_c,
        "finite_cost": bool(math.isfinite(init_c) and math.isfinite(opt_c)),
        "action_sequence": actions.tolist(),
        "action_finite": bool(torch.isfinite(actions).all()),
        "k": int(args.plan_k),
        "horizon": int(args.plan_horizon),
        "env_step": None,
    }
    if args.probe_env and original_env_importable():
        adapter = DiverseMazeEnvAdapter.from_stored_trial(
            difficulty="medium", trial_index=0
        )
        obs0 = adapter.reset()
        a0 = actions[0].numpy()
        obs1 = adapter.step_raw(a0)
        payload["env_step"] = {
            "xy0": obs0.xy.tolist(),
            "xy1": obs1.xy.tolist(),
            "reward_finite": math.isfinite(obs1.reward),
            "image_ok": list(obs1.image.shape) == [3, 98, 98],
            "moved": bool(np.linalg.norm(obs1.xy - obs0.xy) > 0),
        }
    if cfg.output_dir:
        write_json(Path(cfg.output_dir) / "plan_smoke.json", payload)
    _print_json(payload)
    return 0 if payload["finite_cost"] else 1


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.mode == "describe-images":
        return cmd_describe_images(args)
    if args.mode == "env-smoke":
        return cmd_env_smoke(args)

    if args.mode == "overfit":
        cfg = overfit_config()
    elif args.mode == "smoke":
        cfg = smoke_config()
    else:
        cfg = Level1TrainConfig()
    cfg = _apply_cli(cfg, args)

    if args.mode in ("train", "overfit", "smoke"):
        if args.mode == "train":
            cfg.require_25maps = True
        return cmd_train(cfg, label=args.mode)
    if args.mode == "diagnose":
        if cfg.max_windows is None:
            cfg.max_windows = 256
        if args.batch_size is None:
            cfg.batch_size = 8
        cfg.shuffle = False
        cfg.drop_last = False
        cfg.num_workers = 0
        return cmd_diagnose(args, cfg)
    if args.mode == "plan-smoke":
        if cfg.max_windows is None:
            cfg.max_windows = 8
        cfg.batch_size = 1
        cfg.shuffle = False
        cfg.drop_last = False
        cfg.num_workers = 0
        return cmd_plan_smoke(args, cfg)
    raise SystemExit(f"unknown mode {args.mode}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(main())
