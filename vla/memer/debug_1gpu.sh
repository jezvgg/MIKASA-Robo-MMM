#!/usr/bin/env bash
# MemER debug on ONE H100 over all six tasks (owner 2026-10-22): a pipeline check with a large LR
# and few steps, not the paper's recipe. A finished stage is skipped when the script runs again.
#   export (CPU, in the background from the start): the high level's SFT data for episodes 0-19 of
#          every task, the authors' generate_sft_data.py with the full run's settings (192-px views);
#   low:   MemER low level, config memer_lowlevel_debug_1xh100 (episodes 0-979 of every task,
#          batch 32, 2000 steps, peak LR 1e-4), then tests/check_memer_action_error.py on the
#          held-out episodes 980-999 against "hold still";
#   high:  MemER high level, the authors' Qwen3-VL command on the six exports (per-device 4 x their
#          accumulation 8, 400 steps, LR 1e-4 instead of 6e-5), then the authors' deploy-style
#          offline rollout (memer scripts/eval_subtask_rollout.py: every frame, the model's own
#          keyframes as memory) on one held-out episode per task ($HELD_OUT, default 990).
# usage: nohup vla/memer/debug_1gpu.sh [work dir, default ~/memer-debug] > log 2>&1 < /dev/null &
# The short report: <work dir>/REPORT.txt
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
W=${1:-$HOME/memer-debug}
TASKS="samedrawer seasondish cabinetsearch depthrecall takeitback veggie"
DATA=${MEMER_DATA_ROOT:-$HOME/datasets/memer}
LAYOUT=$REPO/vla/memer/configs/fetch_stacked_rgb_camera_layout.json
HELD_OUT=${HELD_OUT:-990}
LOGS=$W/logs REPORT=$W/REPORT.txt
mkdir -p "$LOGS" "$W/sft"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$REPORT"; }

export_sft() {
  for t in $TASKS; do
    [ -f "$W/sft/$t/train.json" ] && continue
    rm -rf "$W/sft/$t"
    (cd "$REPO/third_party/memer" && OMP_NUM_THREADS=2 "$HOME/envs/memer/bin/python" scripts/generate_sft_data.py \
      --lerobot_path "$DATA/$t" --output_dir "$W/sft/$t" --camera_layout_config "$LAYOUT" \
      --keyframe_rule_file "$DATA/$t/keyframe_rules.json" --frame_subsample 5 --recent_frames_length 8 \
      --keyframes_length 8 --prediction_horizon 2 --view_width 192 --view_height 192 \
      --num_workers 2 --max_episodes 20) > "$LOGS/export_$t.log" 2>&1 || echo "export $t FAILED" >> "$LOGS/export.status"
  done
  echo DONE >> "$LOGS/export.status"
}

low() {
  # shellcheck disable=SC1091
  . "$HOME/mikasa-pi05/server.env"
  local py=$OPENPI_DIR/.venv/bin/python cfg=memer_lowlevel_debug_1xh100 exp=debug1
  local ck=$PI05_WORK/checkpoints/$cfg/$exp
  export PYTHONPATH=$REPO/vla/pi05_first_frame/src MEMER_DATA_ROOT=$DATA
  if [ ! -f "$ck/.finished" ]; then
    say "low: training $cfg (log $LOGS/low_train.log)"
    if ! "$py" -m mikasa_pi05 train "$cfg" --exp-name "$exp" --overwrite > "$LOGS/low_train.log" 2>&1; then
      if grep -qE 'RESOURCE_EXHAUSTED|Out of memory|out of memory' "$LOGS/low_train.log"; then
        say "low: out of GPU memory at batch 32; retrying at batch 16"
        "$py" -m mikasa_pi05 train "$cfg" --exp-name "$exp" --overwrite --batch-size 16 > "$LOGS/low_train.log" 2>&1 \
          || { say "low: training FAILED"; tail -n 30 "$LOGS/low_train.log" >> "$REPORT"; return 1; }
      else
        say "low: training FAILED"; tail -n 30 "$LOGS/low_train.log" >> "$REPORT"; return 1
      fi
    fi
    touch "$ck/.finished"
  fi
  say "low: training done; first and last logged steps:"
  grep -oE 'Step [0-9]+: .*' "$LOGS/low_train.log" | sed -n '1p;$p' | sed 's/^/  /' >> "$REPORT"
  local ckpt
  ckpt=$(find "$ck" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' | sort -V | tail -1)
  say "low: open-loop check of $ckpt (log $LOGS/low_check.log)"
  if "$py" "$REPO/vla/pi05_first_frame/tests/check_memer_action_error.py" "$cfg" "$ckpt" 50 \
      > "$LOGS/low_check.log" 2>&1; then
    grep -E '^[a-z]+ +(held-out|training)|^example prompt|frames per task' "$LOGS/low_check.log" | sed 's/^/  /' >> "$REPORT"
  else
    say "low: check FAILED"; tail -n 30 "$LOGS/low_check.log" >> "$REPORT"
  fi
}

high() {
  while ! grep -q DONE "$LOGS/export.status" 2>/dev/null; do sleep 30; done
  grep FAILED "$LOGS/export.status" >> "$REPORT" 2>/dev/null
  local uses="" out=$W/qwen_debug t
  for t in $TASKS; do [ -f "$W/sft/$t/train.json" ] && uses="${uses:+$uses,}memer_$t"; done
  if [ ! -f "$out/.finished" ]; then
    say "high: training on $uses (log $LOGS/high_train.log)"
    (cd "$REPO/third_party/Qwen3-VL/qwen-vl-finetune" && MEMER_SFT_ROOT=$W/sft QWEN_VL_ATTN_IMPL=flash_attention_2 \
      "$HOME/envs/qwen3vl/bin/torchrun" --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29514 \
      qwenvl/train/train_qwen.py --model_name_or_path Qwen/Qwen3-VL-4B-Instruct --dataset_use "$uses" \
      --data_flatten True --tune_mm_vision False --tune_mm_mlp False --tune_mm_llm True --bf16 \
      --output_dir "$out" --num_train_epochs 15 --max_steps 400 --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 8 --optim adamw_torch --max_pixels 115200 --min_pixels 50176 \
      --eval_strategy no --save_strategy no --learning_rate 1e-4 --weight_decay 0 --warmup_ratio 0.05 \
      --max_grad_norm 1 --lr_scheduler_type cosine --logging_steps 1 --logging_nan_inf_filter False \
      --model_max_length 8192 --gradient_checkpointing True --dataloader_num_workers 4 --report_to none) \
      > "$LOGS/high_train.log" 2>&1
    if [ ! -f "$out/config.json" ]; then
      say "high: training FAILED"; tr '\r' '\n' < "$LOGS/high_train.log" | tail -n 30 >> "$REPORT"; return 1
    fi
    touch "$out/.finished"
  fi
  say "high: training done; loss at steps 1, 10, 100, 200, 400:"
  tr '\r' '\n' < "$LOGS/high_train.log" | grep -oE "'loss': [0-9.e+-]+" | awk 'NR==1||NR==10||NR==100||NR==200||NR==400' \
    | tr '\n' ' ' | sed 's/^/  /' >> "$REPORT"; echo >> "$REPORT"
  for t in $TASKS; do
    local ev=$W/high_eval/$t
    if [ ! -f "$ev/summary.json" ]; then
      (cd "$REPO/third_party/memer" && "$HOME/envs/memer/bin/python" scripts/eval_subtask_rollout.py \
        --model-path "$out" --output-dir "$ev" --lerobot-path "$DATA/$t" --camera-layout-config "$LAYOUT" \
        --view-width 192 --view-height 192 --episode-indices "$HELD_OUT" --device cuda --dtype bfloat16 \
        --attn-implementation sdpa) > "$LOGS/high_eval_$t.log" 2>&1 \
        || { say "high: eval $t FAILED"; tail -n 20 "$LOGS/high_eval_$t.log" >> "$REPORT"; continue; }
    fi
    python3 - "$ev" "$t" <<'EOF' | tee -a "$REPORT"
import collections, json, pathlib, sys
ev, task = pathlib.Path(sys.argv[1]), sys.argv[2]
m = json.loads((ev / "summary.json").read_text())["metrics"]
rows = [json.loads(line) for line in (ev / "predictions.jsonl").open()]
majority = collections.Counter(r["target_subtask"] for r in rows).most_common(1)[0][1] / max(len(rows), 1)
keyframes = sum(bool(r["mapped_keyframe_indices"]) for r in rows)
print(f"  {task:14s} frames {m['total_examples']:5d}  subtask accuracy {m['normalized_accuracy']:.3f} "
      f"(raw {m['raw_accuracy']:.3f}; majority-label baseline {majority:.3f})  parse failures "
      f"{m['parse_failure_rate']:.3f}  frames that added a keyframe {keyframes}  memory at the end "
      f"{len(rows[-1]['memory_indices_after']) if rows else 0}")
EOF
  done
}

say "MemER debug on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); repo $(git -C "$REPO" rev-parse --short HEAD)"
export_sft &
low || say "low: stage failed, continuing with the high level"
high || say "high: stage failed"
wait
say "disk: $(du -s --block-size=1G "$HOME" 2>/dev/null | cut -f1) GB used in $HOME"
say "finished"
