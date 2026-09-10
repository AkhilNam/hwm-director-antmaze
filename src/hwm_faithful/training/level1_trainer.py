"""Level-1 trainer for the faithful HWM.

Independently implemented. Matches the executed original loop in
``pldm/train.py`` for L1-only Diverse Maze:

    transpose batch→time, Cosine/Adam step, encode GT, recursive f_L,
    sum VICRegObs + IDM + PredictionProprio, backward, optimizer.step.

Does not copy ``Trainer`` wholesale. Does not train Level 2.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from hwm_faithful.losses.level1_objective import Level1Objective, Level1ObjectiveOutput
from hwm_faithful.models.level1_world_model import Level1WorldModel
from hwm_faithful.training.checkpoint import (
    apply_checkpoint,
    checkpoint_filename,
    load_checkpoint,
    pick_latest_checkpoint,
    save_checkpoint,
)
from hwm_faithful.training.config import ADAM_WEIGHT_DECAY, Level1TrainConfig
from hwm_faithful.training.dataset import make_level1_loader
from hwm_faithful.training.scheduler import CosineWarmupScheduler


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Match original ``seed_everything`` (warn_only deterministic algorithms)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _module_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            return float("nan")
        total += float(g.float().norm().item() ** 2)
    return math.sqrt(total)


def _grads_finite(module: torch.nn.Module) -> bool:
    for p in module.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


def _metrics_from_loss(out: Level1ObjectiveOutput) -> dict[str, float]:
    return {
        "loss": float(out.total.item()),
        "vicreg_obs": float(out.vicreg_obs.item()),
        "vicreg_sim": float(out.vicreg_sim.item()),
        "vicreg_std": float(out.vicreg_std.item()),
        "vicreg_cov": float(out.vicreg_cov.item()),
        "vicreg_std_t": float(out.vicreg_std_t.item()),
        "idm": float(out.idm.item()),
        "idm_action": float(out.idm_action.item()),
        "prediction_proprio": float(out.prediction_proprio.item()),
        "prediction_proprio_raw": float(out.prediction_proprio_raw.item()),
    }


class Level1Trainer:
    """Owns encoder, predictor, IDM, optimizer, cosine scheduler."""

    def __init__(self, cfg: Level1TrainConfig, loader: DataLoader | None = None) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        seed_everything(cfg.seed, deterministic=cfg.deterministic)

        self.model = Level1WorldModel().to(self.device)
        self.objective = Level1Objective().to(self.device)
        if cfg.compile_model:
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        param_groups = [
            {"params": list(self.model.parameters()), "lr": cfg.base_lr},
        ]
        if cfg.include_idm_in_optimizer:
            param_groups.append(
                {"params": list(self.objective.parameters()), "lr": cfg.base_lr}
            )
        self.optimizer = torch.optim.Adam(
            param_groups, weight_decay=cfg.weight_decay or ADAM_WEIGHT_DECAY
        )

        self.loader = loader if loader is not None else make_level1_loader(cfg)
        batch_steps = max(len(self.loader), 1)
        self.scheduler = CosineWarmupScheduler.from_config(
            self.optimizer, cfg, batch_steps=batch_steps
        )

        self.epoch = 0
        self.global_step = 0
        self.sample_step = 0
        self.history: list[dict[str, Any]] = []
        self.output_dir = Path(cfg.output_dir) if cfg.output_dir else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._maybe_resume()

    @property
    def encoder(self) -> torch.nn.Module:
        return self.model.encoder

    @property
    def predictor(self) -> torch.nn.Module:
        return self.model.predictor

    @property
    def idm(self) -> torch.nn.Module:
        return self.objective.idm

    def _log_path(self) -> Path | None:
        if self.output_dir is None:
            return None
        return self.output_dir / "metrics.jsonl"

    def _append_log(self, row: dict[str, Any]) -> None:
        self.history.append(row)
        path = self._log_path()
        if path is None:
            return
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def _maybe_resume(self) -> bool:
        path = None
        if self.cfg.resume_path:
            path = Path(self.cfg.resume_path)
        elif self.cfg.resume_if_possible and self.output_dir is not None:
            path = pick_latest_checkpoint(self.output_dir)
        if path is None:
            return False
        blob = load_checkpoint(path, map_location=self.device)
        apply_checkpoint(
            blob,
            encoder=self.encoder,
            predictor=self.predictor,
            idm=self.idm,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.epoch = int(blob["epoch"])
        self.global_step = int(blob["global_step"])
        self.sample_step = int(blob["sample_step"])
        print(
            f"resumed from {path} epoch={self.epoch} "
            f"step={self.global_step} sample_step={self.sample_step}",
            flush=True,
        )
        return True

    def save(self, epoch: int | None = None) -> Path | None:
        if self.output_dir is None:
            return None
        epoch = self.epoch if epoch is None else int(epoch)
        name = checkpoint_filename(epoch, self.sample_step)
        path = save_checkpoint(
            self.output_dir / name,
            encoder=self.encoder,
            predictor=self.predictor,
            idm=self.idm,
            optimizer=self.optimizer,
            scheduler_state=self.scheduler.state_dict(),
            epoch=epoch,
            global_step=self.global_step,
            sample_step=self.sample_step,
            config=self.cfg,
            seed=self.cfg.seed,
        )
        latest = self.output_dir / "latest.pt"
        shutil.copy2(path, latest)
        print(f"saved {path}", flush=True)
        return path

    def _forward_batch(self, batch: dict[str, torch.Tensor]) -> Level1ObjectiveOutput:
        images = batch["images"].to(self.device, non_blocking=True).transpose(0, 1)
        proprio = batch["proprio"].to(self.device, non_blocking=True).transpose(0, 1)
        actions = batch["actions"].to(self.device, non_blocking=True).transpose(0, 1)
        encoded, rolled = self.model.encode_and_rollout(images, proprio, actions)
        return self.objective(
            obs_gt=encoded.visual,
            obs_pred=rolled.obs_component,
            proprio_gt=encoded.proprio_map,
            proprio_pred=rolled.proprio_component,
            fused_gt=encoded.fused,
            actions=actions,
        )

    def train(self) -> list[dict[str, Any]]:
        cfg = self.cfg
        self.model.train()
        self.objective.train()
        n_passes = cfg.n_epoch_passes
        start_epoch = self.epoch
        steps_done_this_run = 0
        epoch = start_epoch

        while True:
            self.epoch = epoch
            epoch_t0 = time.time()
            n_batches = max(len(self.loader), 1)
            for batch_idx, batch in enumerate(self.loader):
                step = epoch * n_batches + batch_idx
                self.global_step = step
                t0 = time.time()
                lr = self.scheduler.adjust_learning_rate(step)
                bsz = int(batch["images"].shape[0])
                self.sample_step += bsz

                self.optimizer.zero_grad(set_to_none=True)
                out = self._forward_batch(batch)
                loss = out.total
                if not torch.isfinite(loss):
                    raise RuntimeError(f"NaN/inf loss at step {step}: {loss}")
                loss.backward()

                encoder_gn = _module_grad_norm(self.encoder)
                predictor_gn = _module_grad_norm(self.predictor)
                idm_gn = _module_grad_norm(self.idm)
                if not (
                    _grads_finite(self.encoder)
                    and _grads_finite(self.predictor)
                    and _grads_finite(self.idm)
                ):
                    raise RuntimeError(f"non-finite gradients at step {step}")
                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters())
                        + list(self.objective.parameters()),
                        cfg.grad_clip,
                    )
                self.optimizer.step()
                steps_done_this_run += 1

                row = {
                    "epoch": epoch,
                    "step": step,
                    "sample_step": self.sample_step,
                    "lr": float(lr),
                    "batch_size": bsz,
                    "grad_norm_encoder": encoder_gn,
                    "grad_norm_predictor": predictor_gn,
                    "grad_norm_idm": idm_gn,
                    "grad_norm": math.sqrt(
                        encoder_gn**2 + predictor_gn**2 + idm_gn**2
                    ),
                    "train_time": time.time() - t0,
                    **_metrics_from_loss(out),
                }
                if self.device.type == "cuda":
                    row["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
                    row["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved() / 1e6
                if cfg.log_every <= 1 or step % cfg.log_every == 0:
                    self._append_log(row)
                    print(
                        f"epoch={epoch} step={step} loss={row['loss']:.4f} "
                        f"vic={row['vicreg_obs']:.4f} idm={row['idm']:.4f} "
                        f"prop={row['prediction_proprio']:.4f} lr={lr:.6g} "
                        f"gn={row['grad_norm']:.4g}",
                        flush=True,
                    )

                if cfg.max_steps is not None and steps_done_this_run >= cfg.max_steps:
                    self.save(epoch)
                    return self.history

            print(
                f"epoch {epoch} done in {time.time() - epoch_t0:.1f}s "
                f"n_batches={n_batches}",
                flush=True,
            )
            if (epoch > 0 and epoch % cfg.save_every_n_epochs == 0) or epoch >= (
                n_passes - 1
            ):
                self.save(epoch)

            epoch += 1
            if epoch >= n_passes and cfg.max_steps is None:
                break
            # If max_steps is set, keep cycling the loader until it is hit.

        return self.history


def summarize_overfit(history: list[dict[str, Any]]) -> dict[str, Any]:
    if len(history) < 2:
        raise ValueError("need at least two logged steps")
    first, last = history[0], history[-1]
    keys = (
        "loss",
        "vicreg_obs",
        "vicreg_sim",
        "prediction_proprio",
        "idm",
    )
    out: dict[str, Any] = {
        "n_steps": len(history),
        "first": {k: first[k] for k in first if k in keys or k == "loss"},
        "last": {k: last[k] for k in last if k in keys or k == "loss"},
        "nan": any(
            not math.isfinite(float(r.get("loss", float("nan")))) for r in history
        ),
    }
    for k in keys:
        out[f"{k}_decreased"] = float(last[k]) < float(first[k])
    return out
