#!/usr/bin/env bash
# Evaluate one checkpoint on all validation seeds with one policy server per GPU.
#
#   . ~/mikasa-pi05/server.env
#   eval_parallel.sh CONFIG CHECKPOINT_DIR OUT_DIR [GPU_LIST] [extra eval_samedrawer.py args...]
#   e.g. eval_parallel.sh pi05_sd_ff_4xh100 $PI05_WORK/checkpoints/pi05_sd_ff_4xh100/run1/29999 \
#            $PI05_WORK/results/run1-29999 0,1,2,3
#
# GPU i runs a server on port ${PORT_BASE:-8100}+i and a simulator shard i/N rendering on the same GPU.
# SERVER_MEM_FRACTION (default 0.5) caps each server's share of its GPU; lower it next to training.
# Shards resume from their episodes.jsonl, so an interrupted run can be restarted as is.
set -euo pipefail

CONFIG="$1"; CKPT="$2"; OUT="$3"; GPUS="${4:-0}"; shift $(( $# < 4 ? $# : 4 ))
: "${OPENPI_DIR:?source server.env first}" "${SIM_VENV:?}" "${REPO_DIR:?}" "${SAMEDRAWER_DATASET_DIR:?}"
IFS=, read -r -a GPU_IDS <<< "$GPUS"
N=${#GPU_IDS[@]}
mkdir -p "$OUT/logs"

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT

for i in "${!GPU_IDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" XLA_PYTHON_CLIENT_MEM_FRACTION="${SERVER_MEM_FRACTION:-0.5}" \
    "$OPENPI_DIR/.venv/bin/python" -m mikasa_pi05 serve "$CONFIG" --checkpoint "$CKPT" --port $((${PORT_BASE:-8100} + i)) \
    > "$OUT/logs/server-$i.log" 2>&1 &
  pids+=($!)
done

shards=()
for i in "${!GPU_IDS[@]}"; do
  (cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" "$SIM_VENV/bin/python" -W ignore \
    vla/pi05_first_frame/eval_samedrawer.py --server "127.0.0.1:$((${PORT_BASE:-8100} + i))" --shard "$i/$N" \
    --dataset-dir "$SAMEDRAWER_DATASET_DIR" --out "$OUT/shard-$i" "$@" > "$OUT/logs/eval-$i.log" 2>&1) &
  shards+=($!)
done

status=0
for pid in "${shards[@]}"; do wait "$pid" || status=1; done
"$SIM_VENV/bin/python" "$REPO_DIR/vla/pi05_first_frame/eval_samedrawer.py" --summarize "$OUT"/shard-* \
  | tee "$OUT/summary.json"
exit $status
