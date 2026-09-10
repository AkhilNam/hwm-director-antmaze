#!/usr/bin/env python3
"""Render maze2d_large_diverse_25maps into images.npy using original HWM code.

Scientific path (README):
    render_data.py  →  PNG  →  postprocess_images.py  →  images.npy

Cluster constraint:
    scratch inode quota is 1,000,000 files. 4,540,859 PNGs cannot be stored.
    44959 episodes is prime, so official --workers_num cannot split except 1
    or 44959 workers.

This adapter therefore:
    calls the SAME drawer.render_state + select_transforms as the original
    scripts, and writes the postprocessed 98x98 uint8 frames into a memmap
    npy in pickle order. It does not change lookat, crop, resize, maps, or
    trajectories.

Original sources (read-only):
    pldm_envs/diverse_maze/data_generation/render_data.py
    pldm_envs/diverse_maze/data_generation/postprocess_images.py
    pldm_envs/diverse_maze/maze_draw.py
    pldm_envs/diverse_maze/transforms.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pldm_envs.diverse_maze import ant_draw, maze_draw
from pldm_envs.diverse_maze.transforms import select_transforms


EXPECTED_FRAMES = 4_540_859
IMAGE_HW = 98
PROGRESS_SUFFIX = ".progress.json"


def _load_pickle(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _raw_to_npy_frame(raw: np.ndarray, transform, png_roundtrip: bool) -> np.ndarray:
    """Match postprocess_images.process_image on a render_state array."""
    img = Image.fromarray(np.uint8(raw))
    if png_roundtrip:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            img.save(tmp.name, format="png")
            with Image.open(tmp.name) as reopened:
                img = transform(reopened)
                arr = np.array(img)
    else:
        img = transform(img)
        arr = np.array(img)
    if arr.shape != (IMAGE_HW, IMAGE_HW, 3) or arr.dtype != np.uint8:
        raise ValueError(f"unexpected frame {arr.shape} {arr.dtype}")
    return arr


def _make_drawer(config: dict, map_metadata, map_idx: int):
    env = ant_draw.load_environment(
        name=f"{config['env']}_{map_idx}", map_key=map_metadata[map_idx]
    )
    return maze_draw.create_drawer(env, env.name)


def cmd_parity(args: argparse.Namespace) -> int:
    probe = Path(args.probe_path)
    splits = _load_pickle(probe / "data.p")
    config = _load_pickle(probe / "metadata.pt")
    maps = _load_pickle(probe / "train_maps.pt")
    npy = np.load(probe / "images.npy", mmap_mode="r")
    transform = select_transforms(config["env"])

    pairs = []
    n_ep = min(len(splits), 10)
    for e in range(n_ep):
        t_len = len(splits[e]["observations"])
        for t in (0, t_len // 2, t_len - 1):
            pairs.append((e, t))
    # a few extra scattered frames
    for e, t in ((0, 1), (0, 10), (3, 25), (7, 80), (9, 100)):
        if e < len(splits) and t < len(splits[e]["observations"]):
            pairs.append((e, t))
    # unique preserve order
    seen = set()
    uniq = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    cum = np.cumsum([len(s["observations"]) for s in splits])
    drawers = {}
    diffs = []
    rt_diffs = []
    exact = 0
    exact_rt = 0
    t0 = time.time()
    for e, t in uniq:
        map_idx = int(splits[e]["map_idx"])
        if map_idx not in drawers:
            drawers[map_idx] = _make_drawer(config, maps, map_idx)
        raw = drawers[map_idx].render_state(splits[e]["observations"][t])
        direct = _raw_to_npy_frame(raw, transform, png_roundtrip=False)
        roundtrip = _raw_to_npy_frame(raw, transform, png_roundtrip=True)
        g = int(cum[e - 1]) + t if e > 0 else t
        ref = np.asarray(npy[g])
        d = np.abs(direct.astype(np.int16) - ref.astype(np.int16))
        dr = np.abs(roundtrip.astype(np.int16) - ref.astype(np.int16))
        diffs.append(d)
        rt_diffs.append(dr)
        if int(d.max()) == 0:
            exact += 1
        if int(dr.max()) == 0:
            exact_rt += 1
        print(
            f"frame e={e} t={t} g={g} direct_max={int(d.max())} "
            f"png_rt_max={int(dr.max())} raw={tuple(np.asarray(raw).shape)}",
            flush=True,
        )

    stacked = np.stack(diffs)
    stacked_rt = np.stack(rt_diffs)
    report = {
        "n_frames": len(uniq),
        "exact_direct": exact,
        "exact_png_roundtrip": exact_rt,
        "direct_max_abs": int(stacked.max()),
        "direct_mean_abs": float(stacked.mean()),
        "png_roundtrip_max_abs": int(stacked_rt.max()),
        "png_roundtrip_mean_abs": float(stacked_rt.mean()),
        "elapsed_s": time.time() - t0,
        "ref_npy": str(probe / "images.npy"),
        "ref_shape": list(npy.shape),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps(report, indent=2) + "\n")
    # Pass if PNG-roundtrip matches probe exactly (official path).
    if exact_rt != len(uniq):
        print("PARITY FAIL: official PNG+transform does not match probe images.npy", file=sys.stderr)
        return 2
    return 0


def _progress_path(npy_path: Path, worker_id: int | None = None) -> Path:
    if worker_id is None:
        return Path(str(npy_path) + PROGRESS_SUFFIX)
    return Path(str(npy_path) + f".progress.w{worker_id}.json")


def cmd_render(args: argparse.Namespace) -> int:
    data = Path(args.data_path)
    splits = _load_pickle(data / "data.p")
    config = _load_pickle(data / "metadata.pt")
    maps = _load_pickle(data / "train_maps.pt")
    transform = select_transforms(config["env"])
    n_ep_all = len(splits)
    n_frames_all = sum(len(s["observations"]) for s in splits)
    if args.max_episodes is not None:
        n_ep_all = min(n_ep_all, int(args.max_episodes))
        n_frames_all = sum(len(s["observations"]) for s in splits[:n_ep_all])

    range_start = int(args.start_episode or 0)
    range_end = int(args.end_episode) if args.end_episode is not None else n_ep_all
    range_end = min(range_end, n_ep_all)
    if range_start < 0 or range_start >= range_end:
        raise SystemExit(f"bad episode range [{range_start}, {range_end})")

    print(
        f"episodes_total={n_ep_all} frames_total={n_frames_all} "
        f"shard=[{range_start},{range_end}) worker={args.worker_id}",
        flush=True,
    )

    out = Path(args.out_npy)
    out.parent.mkdir(parents=True, exist_ok=True)
    prog_path = _progress_path(out, args.worker_id)
    start_ep = range_start
    if out.is_file() and not args.overwrite:
        arr = np.load(out, mmap_mode="r+")
        print(f"opened existing {arr.shape} {out}", flush=True)
        if tuple(arr.shape) != (n_frames_all, IMAGE_HW, IMAGE_HW, 3):
            raise SystemExit(
                f"existing npy shape {arr.shape} != {(n_frames_all, IMAGE_HW, IMAGE_HW, 3)}"
            )
        if prog_path.is_file():
            prog = json.loads(prog_path.read_text())
            start_ep = max(start_ep, int(prog.get("last_completed_episode", -1)) + 1)
            print(f"resume shard from episode {start_ep}", flush=True)
        else:
            legacy = Path(str(out) + PROGRESS_SUFFIX)
            if legacy.is_file() and range_start == 0:
                prog = json.loads(legacy.read_text())
                start_ep = max(start_ep, int(prog.get("last_completed_episode", -1)) + 1)
                print(f"resume from legacy progress episode {start_ep}", flush=True)
    else:
        print(f"creating memmap {out} shape={(n_frames_all, IMAGE_HW, IMAGE_HW, 3)}", flush=True)
        arr = np.lib.format.open_memmap(
            out, mode="w+", dtype=np.uint8, shape=(n_frames_all, IMAGE_HW, IMAGE_HW, 3)
        )
        arr.flush()

    if start_ep >= range_end:
        print(f"shard already complete [{range_start},{range_end})", flush=True)
        return 0

    png_rt = bool(args.png_roundtrip)
    drawers = {}
    t0 = time.time()
    written = 0
    offset = sum(len(s["observations"]) for s in splits[:start_ep])
    for e in range(start_ep, range_end):
        split = splits[e]
        map_idx = int(split["map_idx"])
        if map_idx not in drawers:
            drawers[map_idx] = _make_drawer(config, maps, map_idx)
        obs = split["observations"]
        for t in range(len(obs)):
            raw = drawers[map_idx].render_state(obs[t])
            arr[offset + t] = _raw_to_npy_frame(raw, transform, png_rt)
        offset += len(obs)
        written += len(obs)
        prog_path.write_text(
            json.dumps(
                {
                    "worker_id": args.worker_id,
                    "last_completed_episode": e,
                    "shard": [range_start, range_end],
                    "frames_done_global": offset,
                    "frames_written_this_run": written,
                    "n_frames": n_frames_all,
                    "elapsed_s": time.time() - t0,
                }
            )
            + "\n"
        )
        if (e + 1) % 10 == 0 or e == start_ep:
            rate = written / max(time.time() - t0, 1e-6)
            remain = sum(len(s["observations"]) for s in splits[e + 1 : range_end])
            eta = remain / max(rate, 1e-6)
            print(
                f"w={args.worker_id} episode {e}/{range_end-1} "
                f"written={written} rate={rate:.2f}/s eta_h={eta/3600:.2f}",
                flush=True,
            )
        if args.flush_every and (e + 1) % int(args.flush_every) == 0:
            arr.flush()

    arr.flush()
    print(
        f"DONE shard [{range_start},{range_end}) npy={out} "
        f"shape={arr.shape} bytes={out.stat().st_size}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("parity", "render"), required=True)
    p.add_argument("--probe-path", default=None)
    p.add_argument("--data-path", default=None)
    p.add_argument("--out-npy", default=None)
    p.add_argument("--report-json", default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--start-episode", type=int, default=0)
    p.add_argument("--end-episode", type=int, default=None)
    p.add_argument("--worker-id", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--png-roundtrip", action="store_true")
    p.add_argument("--flush-every", type=int, default=50)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "parity":
        if not args.probe_path:
            raise SystemExit("--probe-path required")
        return cmd_parity(args)
    if not args.data_path or not args.out_npy:
        raise SystemExit("--data-path and --out-npy required")
    return cmd_render(args)


if __name__ == "__main__":
    raise SystemExit(main())
