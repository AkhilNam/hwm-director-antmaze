#!/bin/bash
# Official-path 25-map image corpus for faithful L1:
#   original drawer.render_state + CenterCrop(426)+Resize(98)
# Writes images.npy directly because scratch inode quota (1e6) cannot
# hold 4.54M PNG files. Does not modify original renderer source.
#
# 44959 episodes is prime: official --workers_num cannot split except
# 1 or 44959 workers. This job is one EGL worker with resume.
#
#SBATCH -A gts-awu36-paid
#SBATCH -q inferno
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 3-00:00:00
#SBATCH -J hwm-25maps-render
#SBATCH -o /storage/scratch1/2/anampally3/hwm-director-antmaze/logs/hwm-25maps-render-%j.out
#SBATCH -e /storage/scratch1/2/anampally3/hwm-director-antmaze/logs/hwm-25maps-render-%j.err

set -euo pipefail

echo "START 25maps render $(date) host=$(hostname) job=${SLURM_JOB_ID:-none}"

export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export PYTHONUNBUFFERED=1

UNIFIED=/storage/scratch1/2/anampally3/hwm-director-antmaze
ORIG=/storage/scratch1/2/anampally3/hwm-original-repro
PY=/storage/scratch1/2/anampally3/envs/hwm-repro/bin/python
PROBE="$ORIG/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe"
DATA="$ORIG/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps"
SCRIPT="$UNIFIED/scripts/render_hwm_25maps_to_npy.py"
LOGDIR="$UNIFIED/logs"

export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="${ORIG}:${PYTHONPATH:-}"
export HWM_ORIGINAL_ROOT="$ORIG"

mkdir -p "$LOGDIR" "$UNIFIED/checkpoints/hwm_faithful/render"

echo "date:      $(date)"
echo "hostname:  $(hostname)"
echo "worker:    single EGL process (44959 is prime; official workers_num cannot split)"
echo "data:      $DATA"
echo "out:       $DATA/images.npy"
nvidia-smi || true

test -x "$PY"
test -f "$SCRIPT"
test -f "$PROBE/images.npy"
test -f "$DATA/data.p"

echo "=== STEP A: probe pixel parity ==="
"$PY" "$SCRIPT" --mode parity \
  --probe-path "$PROBE" \
  --report-json "$UNIFIED/checkpoints/hwm_faithful/render/probe_parity.json"

echo "=== STEP B: 2-episode pilot npy ==="
PILOT="$UNIFIED/checkpoints/hwm_faithful/render/pilot_2ep.npy"
rm -f "$PILOT" "${PILOT}.progress.json"
"$PY" "$SCRIPT" --mode render \
  --data-path "$DATA" \
  --out-npy "$PILOT" \
  --max-episodes 2 \
  --png-roundtrip \
  --overwrite

"$PY" - <<PY
import numpy as np
from pathlib import Path
p=Path("$PILOT")
a=np.load(p, mmap_mode="r")
assert a.shape==(202,98,98,3), a.shape
assert a.dtype==np.uint8, a.dtype
assert a.min()>=0 and a.max()<=255
print("pilot_ok", a.shape, a.dtype, "min", int(a.min()), "max", int(a.max()), "bytes", p.stat().st_size)
PY

echo "=== STEP C: full 25-map images.npy (resume-capable) ==="
"$PY" "$SCRIPT" --mode render \
  --data-path "$DATA" \
  --out-npy "$DATA/images.npy" \
  --png-roundtrip

echo "DONE 25maps render $(date) host=$(hostname)"
ls -lh "$DATA/images.npy" || true
