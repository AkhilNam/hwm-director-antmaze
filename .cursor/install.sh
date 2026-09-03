#!/usr/bin/env bash
# Idempotent Cloud Agent setup for hwm-director-antmaze.
#
# Installs the project and its dev dependencies into the pinned system Python
# (python3.12) so that `python3.12 ...` works in later commands without any
# virtualenv activation. Safe to run repeatedly.
set -euo pipefail

PY=python3.12

# The default image ships python3.12 + pip; guard the rare case they are
# missing so the script also works on a barer base image.
if ! command -v "$PY" >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "$PY"
fi
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends python3-pip
fi

# PEP 668 marks the Debian/Ubuntu system Python as externally managed, so
# --break-system-packages is required to install into it.
"$PY" -m pip install --break-system-packages --upgrade pip
"$PY" -m pip install --break-system-packages -e ".[dev]"

# Best-effort pre-download of the default Minari dataset (D4RL/antmaze/umaze-v1)
# so training/eval scripts start immediately. Non-fatal if the Hugging Face Hub
# is unreachable at build time; the scripts fetch it on first use.
"$PY" - <<'PY' || echo "minari pre-download skipped (dataset will download on first use)"
import minari

minari.load_dataset("D4RL/antmaze/umaze-v1", download=True)
print("minari dataset ready")
PY

echo "hwm-director-antmaze environment ready"
