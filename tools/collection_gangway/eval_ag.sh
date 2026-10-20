#!/usr/bin/env bash
# pi0.5 run4-h50-lowlr on the first 10 validation seeds in the assisted-grasp environment (profile 21),
# same flags as results/run4-h50-lowlr-final-reset (replan every 25 of 50, reset first frame), videos.
set -u
. $HOME/mikasa-pi05/server.env
source $HOME/sd-collect/env.sh
CKPT=$HOME/mikasa-pi05/checkpoints/pi05_sd_ff_h50_4xh100/run4-h50-lowlr/29999
OUT=$HOME/mikasa-pi05/results/run4-h50-lowlr-ag-first10
mkdir -p $OUT/logs
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 "$OPENPI_DIR/.venv/bin/python" -m mikasa_pi05 serve \
  pi05_sd_ff_h50_4xh100 --checkpoint $CKPT --port 8150 > $OUT/logs/server.log 2>&1 &
SRV=$!; echo $SRV > $OUT/logs/server.pid
trap "kill $SRV 2>/dev/null" EXIT
cd $HOME/sd-collect/eval-ag
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. "$SIM" -W ignore vla/pi05_first_frame/eval_samedrawer.py --server 127.0.0.1:8150 \
  --dataset-dir "$SAMEDRAWER_DATASET_DIR" --out $OUT/shard-0 --seeds validation --max-episodes 10 \
  --first-frame reset --replan-steps 25 --video 10 --allow-signature-mismatch > $OUT/logs/eval-0.log 2>&1
echo "eval exit=$?" >> $OUT/logs/eval-0.log
"$SIM" vla/pi05_first_frame/eval_samedrawer.py --summarize $OUT/shard-0 > $OUT/summary.json 2>> $OUT/logs/eval-0.log
date > $OUT/DONE
