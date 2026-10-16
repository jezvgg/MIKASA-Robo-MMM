#!/usr/bin/env bash
# Run this repository's CPU code on a Mac inside the linux/amd64 planner image.
#
#   utils/docker/amd64.sh python planners/depth_recall_v1_planner.py --seed 0 --assisted-grasp
#   utils/docker/amd64.sh python -m pytest -q utils/mikasa/test_assisted_grasp.py
#
# Why: mplib ships only manylinux x86_64 wheels, so the planners cannot run on macOS.
# Docker Desktop's amd64 emulation can, on the CPU simulator with lavapipe rendering —
# every number from here is "CPU, emulated". The image is `mikasa-planner:amd64`
# (mani_skill 3.0.0b22, sapien 3.0.3, mplib 0.2.1, CPU torch), built from mikasa-bench's
# tools/docker/Dockerfile.planner-amd64 with `tools/docker/planner.sh build`.
# Docker Desktop must be running (`open -a Docker`).
#
# Mounts: this checkout read-write at /work (imported via PYTHONPATH, nothing installed),
# the ManiSkill/RoboCasa assets read-only. MS_ASSET_DIR names the PARENT of `data`
# (mani_skill appends "/data"), and a pycache prefix keeps host bytecode out of the
# container and Linux bytecode out of the host tree.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${MIKASA_PLANNER_IMAGE:-mikasa-planner:amd64}"
ASSETS="${MS_ASSET_DIR_HOST:-$HOME/.maniskill/data}"
[ $# -ge 1 ] || { sed -n '2,6p' "$0"; exit 1; }
[ -d "$ASSETS/scene_datasets/robocasa_dataset" ] || {
  echo "RoboCasa assets not found under $ASSETS (set MS_ASSET_DIR_HOST)" >&2; exit 2; }
tty=()
[ -t 0 ] && [ -t 1 ] && tty=(-it)
exec docker run --rm ${tty[@]+"${tty[@]}"} --platform linux/amd64 \
  -v "$REPO:/work" -v "$ASSETS:/root/.maniskill/data:ro" -w /work \
  -e MS_ASSET_DIR=/root/.maniskill -e PYTHONPYCACHEPREFIX=/tmp/pycache -e PYTHONPATH=/work \
  "$IMAGE" "$@"
