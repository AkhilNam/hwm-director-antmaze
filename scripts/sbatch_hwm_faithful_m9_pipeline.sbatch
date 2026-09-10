#!/bin/bash
# M9 pipeline on a GPU node:
#   1) M8 env reset + one step (closes the login-node GL gap)
#   2) tiny real-data L1 overfit on probe images
#   3) short L1 GPU smoke (~300 steps) on probe
#   4) diagnostics + optional flat L1 MPPI smoke
#
# Does NOT train Level 2.
# Does NOT launch full 25-map L1 unless images.npy exists and matches 4.54M frames.
# Do not regenerate maze2d_large_diverse_25maps/images.npy.
#
#SBATCH -A gts-awu36-paid
#SBATCH -q inferno
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J hwm-faithful-m9
#SBATCH -o /storage/scratch1/2/anampally3/hwm-director-antmaze/logs/hwm-faithful-m9-%j.out
#SBATCH -e /storage/scratch1/2/anampally3/hwm-director-antmaze/logs/hwm-faithful-m9-%j.err

set -euo pipefail

echo "START faithful M9 pipeline $(date) host=$(hostname)"

export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/storage/scratch1/2/anampally3/hwm-director-antmaze
ORIG=/storage/scratch1/2/anampally3/hwm-original-repro
ENV=/storage/scratch1/2/anampally3/envs/hwm-repro
PY="$ENV/bin/python"
PROBE="$ORIG/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe"
TRAIN_PKL="$ORIG/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps/data.p"
OUT_ROOT="$REPO/checkpoints/hwm_faithful"
LOGDIR="$REPO/logs"

export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src:${ORIG}:${PYTHONPATH:-}"
export HWM_ORIGINAL_ROOT="$ORIG"

mkdir -p "$LOGDIR" "$OUT_ROOT/l1_overfit" "$OUT_ROOT/l1_smoke" "$OUT_ROOT/l1"

echo "date:      $(date)"
echo "hostname:  $(hostname)"
echo "python:    $(python --version 2>&1)"
nvidia-smi || true

test -x "$PY"
test -f "$PROBE/data.p"
test -f "$PROBE/images.npy"

cd "$REPO"

echo "=== describe 25-map images (must not regenerate) ==="
set +e
"$PY" -m hwm_faithful.training.cli --mode describe-images
DESC_RC=$?
set -e
echo "describe-images exit=$DESC_RC"

echo "=== M8 GPU env reset/step ==="
"$PY" -m hwm_faithful.training.cli --mode env-smoke --output-dir "$OUT_ROOT/l1_overfit"

echo "=== tiny real overfit (probe subset; 25-map images missing) ==="
"$PY" -m hwm_faithful.training.cli --mode overfit \
  --data-path "$PROBE/data.p" \
  --images-path "$PROBE/images.npy" \
  --output-dir "$OUT_ROOT/l1_overfit" \
  --device cuda \
  --no-resume \
  --max-windows 8 \
  --batch-size 4 \
  --max-steps 40

echo "=== diagnose overfit checkpoint ==="
"$PY" -m hwm_faithful.training.cli --mode diagnose \
  --data-path "$PROBE/data.p" \
  --images-path "$PROBE/images.npy" \
  --output-dir "$OUT_ROOT/l1_overfit" \
  --device cuda \
  --max-windows 32 \
  --batch-size 8

echo "=== flat L1 MPPI smoke ==="
"$PY" -m hwm_faithful.training.cli --mode plan-smoke \
  --data-path "$PROBE/data.p" \
  --images-path "$PROBE/images.npy" \
  --output-dir "$OUT_ROOT/l1_overfit" \
  --device cuda \
  --plan-k 32 \
  --plan-horizon 10 \
  --probe-env

echo "=== short GPU training smoke (~300 steps, probe) ==="
"$PY" -m hwm_faithful.training.cli --mode smoke \
  --data-path "$PROBE/data.p" \
  --images-path "$PROBE/images.npy" \
  --output-dir "$OUT_ROOT/l1_smoke" \
  --device cuda \
  --no-resume \
  --num-workers 4 \
  --max-steps 300

echo "=== smoke resume 2 steps ==="
"$PY" -m hwm_faithful.training.cli --mode smoke \
  --data-path "$PROBE/data.p" \
  --images-path "$PROBE/images.npy" \
  --output-dir "$OUT_ROOT/l1_smoke" \
  --device cuda \
  --max-steps 2

IMAGES_25="${HWM_25MAPS_IMAGES:-$ORIG/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps/images.npy}"
if [[ -f "$IMAGES_25" ]]; then
  echo "=== 25-map images found at $IMAGES_25 ; NOT auto-launching full train from this short job ==="
  echo "Submit scripts/sbatch_hwm_faithful_l1_full.sbatch after reviewing smoke metrics."
else
  echo "=== BLOCKER: 25-map images.npy not found. Full L1 training not launched. ==="
fi

echo "DONE faithful M9 pipeline $(date) host=$(hostname)"
