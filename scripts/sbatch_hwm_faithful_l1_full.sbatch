#!/bin/bash
# Full faithful Level-1 training on maze2d_large_diverse_25maps.
# Submit ONLY after M9 env-smoke + overfit + GPU-smoke pass AND images.npy exists.
# Does NOT train Level 2.
# Do not regenerate images.npy.
#
#SBATCH -A gts-awu36-paid
#SBATCH -q inferno
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH -J hwm-faithful-l1
#SBATCH -o /storage/scratch1/2/anampally3/hwm-director-antmaze/logs/hwm-faithful-l1-%j.out
#SBATCH -e /storage/scratch1/2/anampally3/hwm-director-antmaze/logs/hwm-faithful-l1-%j.err

set -euo pipefail

echo "START faithful L1 full $(date) host=$(hostname)"

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
DATA="$ORIG/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps/data.p"
IMAGES="${HWM_25MAPS_IMAGES:-$ORIG/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps/images.npy}"
OUT="$REPO/checkpoints/hwm_faithful/l1"

export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src:${ORIG}:${PYTHONPATH:-}"
export HWM_ORIGINAL_ROOT="$ORIG"

mkdir -p "$OUT" "$REPO/logs"

echo "date:      $(date)"
echo "hostname:  $(hostname)"
nvidia-smi || true

test -x "$PY"
test -f "$DATA"
if [[ ! -f "$IMAGES" ]]; then
  echo "BLOCKER: 25-map images.npy missing at $IMAGES" >&2
  echo "Do not regenerate. Aborting." >&2
  exit 2
fi

cd "$REPO"
"$PY" -m hwm_faithful.training.cli --mode describe-images --images-path "$IMAGES"

"$PY" -m hwm_faithful.training.cli --mode train \
  --data-path "$DATA" \
  --images-path "$IMAGES" \
  --output-dir "$OUT" \
  --device cuda \
  --require-25maps \
  --num-workers 10

echo "DONE faithful L1 full $(date) host=$(hostname)"
