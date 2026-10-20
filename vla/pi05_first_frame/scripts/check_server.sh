#!/usr/bin/env bash
# Quick checks after setup_server.sh (about 5 minutes). Source server.env first.
#   check_server.sh            driver, GPUs, both venvs, dataset reader, simulator + render parity
#   check_server.sh --base     also load pi05_base with the 4-image config and run inference
set -euo pipefail
: "${OPENPI_DIR:?source server.env first}" "${SIM_VENV:?}" "${REPO_DIR:?}" "${SAMEDRAWER_DATASET_DIR:?}"
HERE="$REPO_DIR/vla/pi05_first_frame"
OUT="${PI05_WORK:?}/results/check-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

echo "== GPUs (pick free ones with CUDA_VISIBLE_DEVICES)"
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu --format=csv
if command -v qd-gpucheck >/dev/null 2>&1; then
  echo "== qd-gpucheck (can this node render?)"
  qd-gpucheck | tee "$OUT/qd-gpucheck.log" | grep -E '"verdict"' || true
  if grep -q '"wedged"' "$OUT/qd-gpucheck.log"; then
    echo "this node cannot render; ask the owner to restart the job on another node" >&2; exit 1
  fi
fi

echo "== openpi venv: JAX"
"$OPENPI_DIR/.venv/bin/python" -c "import jax; print(jax.__version__, jax.devices())"

echo "== simulator venv: torch CUDA (sapien_cuda hands camera images to torch on the GPU)"
"$SIM_VENV/bin/python" -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available()); assert torch.cuda.is_available()"

if [[ ! -f "$PI05_WORK/assets/samedrawer/nurtayev-d/samedrawer-1000ep/norm_stats.json" ]]; then
  echo "== norm stats (once per server, ~1 min)"
  "$OPENPI_DIR/.venv/bin/python" -m mikasa_pi05 norm-stats pi05_sd_ff_4xh100 2>&1 | grep -E 'frames in|q01==q99'
fi

echo "== dataset reader and tests"
(cd "$REPO_DIR" && "$OPENPI_DIR/.venv/bin/python" -m pytest "$HERE/tests" -q -p no:cacheprovider)

echo "== simulator: recorded actions of 2 training episodes through the evaluation loop (expect 2/2)"
(cd "$REPO_DIR" && "$SIM_VENV/bin/python" -W ignore "$HERE/eval_samedrawer.py" --policy recorded --seeds train:2 \
  --dataset-dir "$SAMEDRAWER_DATASET_DIR" --out "$OUT/recorded" | tail -n 3)
grep -q '"success_rate": 1.0' "$OUT/recorded/summary.json" || { echo "recorded replay failed" >&2; exit 1; }

echo "== live render vs dataset frame 0 (expect ~1-2 for 'live vs frame0', much larger for the others)"
(cd "$REPO_DIR" && PYTHONPATH=. "$SIM_VENV/bin/python" -W ignore "$HERE/tests/render_parity.py" dump "$OUT/render.npz" 8 \
  && "$OPENPI_DIR/.venv/bin/python" "$HERE/tests/render_parity.py" compare "$OUT/render.npz")

if [[ "${1:-}" == "--base" ]]; then
  echo "== pi05_base with 4 images: load + inference"
  "$OPENPI_DIR/.venv/bin/python" "$HERE/tests/check_base_inference.py"
fi
echo "all checks passed; results in $OUT"
