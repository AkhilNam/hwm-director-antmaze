#!/bin/bash
# Faster 25-map render: 8 parallel EGL workers on L40S.
# Maze rendering is OpenGL rasterization, not GEMM. L40S is the right
# accelerator; 8 workers beat one H100. Shards write non-overlapping
# episode ranges into the already-allocated 122GB images.npy.
#
#SBATCH -A gts-awu36-paid
#SBATCH -q inferno
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 1-00:00:00
#SBATCH -J hwm-25maps-l40s
#SBATCH -o /storage/scratch1/2/anampally3/hwm-director-antmaze/logs/hwm-25maps-l40s-%A_%a.out
#SBATCH -e /storage/scratch1/2/anampally3/hwm-director-antmaze/logs/hwm-25maps-l40s-%A_%a.err

set -euo pipefail

WID="${SLURM_ARRAY_TASK_ID:?}"
NWORK=8
NEP=44959

echo "START shard ${WID}/${NWORK} $(date) host=$(hostname) job=${SLURM_JOB_ID:-none}"

export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export PYTHONUNBUFFERED=1

UNIFIED=/storage/scratch1/2/anampally3/hwm-director-antmaze
ORIG=/storage/scratch1/2/anampally3/hwm-original-repro
PY=/storage/scratch1/2/anampally3/envs/hwm-repro/bin/python
DATA="$ORIG/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps"
SCRIPT="$UNIFIED/scripts/render_hwm_25maps_to_npy.py"

export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="${ORIG}:${PYTHONPATH:-}"
export HWM_ORIGINAL_ROOT="$ORIG"

# Even split; last worker takes the remainder.
START=$(( WID * NEP / NWORK ))
END=$(( (WID + 1) * NEP / NWORK ))

echo "date:     $(date)"
echo "hostname: $(hostname)"
echo "partition:${SLURM_JOB_PARTITION:-} gres=${SLURM_JOB_GRES:-}"
echo "worker:   ${WID}  episodes [${START}, ${END})"
echo "out:      $DATA/images.npy"
nvidia-smi -L || true

test -x "$PY"
test -f "$SCRIPT"
test -f "$DATA/images.npy"

"$PY" "$SCRIPT" --mode render \
  --data-path "$DATA" \
  --out-npy "$DATA/images.npy" \
  --start-episode "$START" \
  --end-episode "$END" \
  --worker-id "$WID"

echo "DONE shard ${WID} $(date) host=$(hostname)"
