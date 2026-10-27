#!/usr/bin/env bash
# Four-GPU debug of the three trainings, through the full runs' own launchers and settings, in
# short runs (the one-GPU debugs, vla/memer/debug_1gpu.sh and vla/memoryvla/debug_1gpu.sh, checked
# that each learns; this checks the four-GPU runs: launch, memory, step time, loss, and with
# CHECKPOINTS=1 the checkpoint and an evaluation from it). Stages, in $STAGES order; a finished
# stage is skipped on a re-run:
#   low   MemER low level (openpi): config memer_lowlevel_debug_4xh100 (the full run's batch 128 on
#         4 FSDP devices, its LR schedule and EMA, episodes 0-979 of every task), $LOW_STEPS steps;
#         with checkpoints, a params-only one at the end and tests/check_memer_action_error.py on
#         the held-out episodes 980-999 against "hold still".
#   high  MemER high level (Qwen3-VL-4B): vla/memer/train_high.sh (per-device 4 x their
#         accumulation 8 x 4 GPUs = 128) on $HIGH_DATASETS under $SFT_ROOT, $HIGH_STEPS steps; with
#         checkpoints, the weights at the end and the authors' eval_subtask_rollout.py from them on
#         episode $HIGH_EVAL of every task.
#   mvla  MemoryVLA: RLDS of episodes 0..$MVLA_EPISODES-1 of every task, then vla/memoryvla/train.sh
#         with the first per-device batch of $MVLA_BS that runs $MVLA_STEPS steps (its memory bank
#         fills over the first 256 frames of an episode, so a batch must outlast that); with
#         checkpoints, one at the end and check_action_error.py on held-out $MVLA_HELD_OUT.
# CHECKPOINTS=0 (default): nothing is saved: every run is set to save past its end and is stopped
# once it has logged its steps. CHECKPOINTS=1 saves and evaluates; it needs ~7 GB (low), ~9 GB
# (high), ~36 GB (mvla) free in our 400 GB, and deletes each checkpoint after its check unless KEEP=1.
# Peak GPU memory (nvidia-smi every 10 s) and host memory (every 30 s) are logged per run.
# usage: nohup vla/debug_4gpu.sh [work dir, default ~/debug4] > log 2>&1 < /dev/null &
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
W=${1:-$HOME/debug4}
STAGES=${STAGES:-low high mvla}
CHECKPOINTS=${CHECKPOINTS:-0}
TASKS="samedrawer seasondish cabinetsearch depthrecall takeitback veggie"
DATA=${MEMER_DATA_ROOT:-$HOME/datasets/memer}
SFT_ROOT=${SFT_ROOT:-$HOME/datasets/memer_sft}
HIGH_DATASETS=${HIGH_DATASETS:-memer_samedrawer,memer_seasondish,memer_cabinetsearch,memer_depthrecall}
LOW_STEPS=${LOW_STEPS:-500}; HIGH_STEPS=${HIGH_STEPS:-60}; HIGH_EVAL=${HIGH_EVAL:-999}
MVLA_EPISODES=${MVLA_EPISODES:-3}; MVLA_STEPS=${MVLA_STEPS:-300}; MVLA_BS=${MVLA_BS:-32 16 8 4 2}
MVLA_HELD_OUT=${MVLA_HELD_OUT:-990,991}
ENDLESS=10000000   # steps and save interval of a run that is stopped from outside
LAYOUT=$REPO/vla/memer/configs/fetch_stacked_rgb_camera_layout.json
LOGS=$W/logs REPORT=$W/REPORT.txt
mkdir -p "$LOGS"
say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$REPORT"; }

watch_start() {  # watch_start NAME: GPU and host memory logs for one run
  nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 10 \
    > "$LOGS/$1.gpu.csv" 2>/dev/null &
  GPU_LOGGER=$!
  (while :; do printf '%s %s\n' "$(date '+%T')" "$(free -g | awk '/Mem:/{print "used", $3, "GB, available", $7, "GB"}')"; sleep 30; done) \
    > "$LOGS/$1.host.log" 2>&1 &
  HOST_LOGGER=$!
}
watch_stop() {  # watch_stop NAME: stop the loggers and report the peaks
  kill "$GPU_LOGGER" "$HOST_LOGGER" 2>/dev/null; wait "$GPU_LOGGER" "$HOST_LOGGER" 2>/dev/null
  awk -F', ' '{gsub(/ MiB/, "", $3); if ($3 > peak[$2]) peak[$2] = $3; if ($4+0 > 0) {busy[$2] += $4; n[$2]++}}
    END {for (g in peak) printf "GPU%s %.1f GB (mean util %d%%)  ", g, peak[g] / 1024, n[g] ? busy[g] / n[g] : 0; print ""}' \
    "$LOGS/$1.gpu.csv" | sed 's/^/  peak GPU memory: /' | tee -a "$REPORT"
  awk '{if ($3 > m) m = $3} END {print "  peak host memory used: " m " GB (machine total, other users included)"}' "$LOGS/$1.host.log" | tee -a "$REPORT"
}

# run_until NAME N COUNTER -- CMD...: run CMD in its own process group (log $LOGS/NAME.log) with
# the loggers on; with CHECKPOINTS=0 stop it once `COUNTER` (a shell snippet printing the number of
# steps logged so far) reaches N. Returns CMD's exit code, or 0 when it was stopped at N.
run_until() {
  local name=$1 n=$2 counter=$3 pid rc=0; shift 4
  watch_start "$name"
  setsid "$@" > "$LOGS/$name.log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$CHECKPOINTS" = 0 ] && [ "$(eval "$counter" 2>/dev/null || echo 0)" -ge "$n" ]; then
      say "$name: $n steps logged; stopping the run"
      kill -TERM -- "-$pid" 2>/dev/null; sleep 40; kill -KILL -- "-$pid" 2>/dev/null
      wait "$pid" 2>/dev/null; rc=0; break
    fi
    sleep 15
  done
  if [ "$CHECKPOINTS" = 1 ] || ! [ "$(eval "$counter" 2>/dev/null || echo 0)" -ge "$n" ]; then wait "$pid"; rc=$?; fi
  watch_stop "$name"
  return $rc
}

low() {
  # shellcheck disable=SC1091
  . "$HOME/mikasa-pi05/server.env"
  local py=$OPENPI_DIR/.venv/bin/python cfg=memer_lowlevel_debug_4xh100 exp=debug4 ckpt steps
  local ck=$PI05_WORK/checkpoints/$cfg/$exp
  export PYTHONPATH=$REPO/vla/pi05_first_frame/src MEMER_DATA_ROOT=$DATA
  [ -f "$W/.low_done" ] && return 0
  steps=$LOW_STEPS; [ "$CHECKPOINTS" = 0 ] && steps=$ENDLESS
  say "low: $cfg on 4 GPUs, $LOW_STEPS steps (log $LOGS/low_train.log)"
  run_until low_train "$LOW_STEPS" "grep -oE 'Step [0-9]+:' '$LOGS/low_train.log' | tail -1 | grep -oE '[0-9]+'" -- \
    env CUDA_VISIBLE_DEVICES=0,1,2,3 "$py" -m mikasa_pi05 train "$cfg" --exp-name "$exp" --overwrite \
      --num-train-steps "$steps" --save-interval "$([ "$CHECKPOINTS" = 0 ] && echo $ENDLESS || echo 200)" \
      --params-only-checkpoints --no-snapshots
  local rc=$?
  [ $rc -eq 0 ] || { say "low: training FAILED (exit $rc)"; tail -n 30 "$LOGS/low_train.log" >> "$REPORT"; return 1; }
  say "low: logged steps:"
  grep -oE 'Step [0-9]+: .*' "$LOGS/low_train.log" | sed 's/^/  /' | tee -a "$REPORT"
  "$py" - "$LOGS/low_train.log" <<'EOF' | tee -a "$REPORT"
import re, sys
t = [float(x) for x in re.findall(r"(\d+\.\d+)s/it", open(sys.argv[1]).read())]
t = sorted(t[len(t) // 5:])  # past compilation
print(f"  median step time {t[len(t) // 2]:.2f} s" if t else "  step time: not in the log")
EOF
  if [ "$CHECKPOINTS" = 1 ]; then
    ckpt=$(find "$ck" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' | sort -V | tail -1)
    say "low: checkpoint $ckpt ($(du -sh "$ckpt" | cut -f1)); open-loop check on held-out 980-999 (log $LOGS/low_check.log)"
    if CUDA_VISIBLE_DEVICES=0 "$py" "$REPO/vla/pi05_first_frame/tests/check_memer_action_error.py" "$cfg" "$ckpt" 50 \
        > "$LOGS/low_check.log" 2>&1; then
      grep -E '^[a-z]+ +(held-out|training)|^example prompt|frames per task' "$LOGS/low_check.log" | sed 's/^/  /' | tee -a "$REPORT"
    else
      say "low: check FAILED"; tail -n 30 "$LOGS/low_check.log" >> "$REPORT"; return 1
    fi
    [ "${KEEP:-0}" = 1 ] || { rm -rf "$ck"; say "low: checkpoint deleted"; }
  fi
  rm -rf "$ck"/[0-9]*   # nothing is saved with CHECKPOINTS=0; leave no partial save behind
  touch "$W/.low_done"
}

high() {
  local out=$W/qwen t ev steps save
  [ -f "$W/.high_done" ] && return 0
  steps=$HIGH_STEPS save=$ENDLESS; [ "$CHECKPOINTS" = 0 ] && steps=$ENDLESS
  rm -rf "$out"
  say "high: train_high.sh on 4 GPUs, $HIGH_DATASETS, $HIGH_STEPS steps (log $LOGS/high_train.log)"
  # Save steps past the end: no checkpoint-* during the run; with checkpoints the final save is checked.
  run_until high_train "$HIGH_STEPS" "tr '\r' '\n' < '$LOGS/high_train.log' | grep -c \"'loss':\"" -- \
    env SAVE_ONLY_MODEL=1 DATASETS="$HIGH_DATASETS" "$REPO/vla/memer/train_high.sh" "$SFT_ROOT" "$out" "$steps" "$save"
  local rc=$?
  [ $rc -eq 0 ] || { say "high: training FAILED (exit $rc)"; tr '\r' '\n' < "$LOGS/high_train.log" | tail -n 30 >> "$REPORT"; return 1; }
  tr '\r' '\n' < "$LOGS/high_train.log" | grep -E "Total training samples" | tail -1 | sed 's/^/  /' | tee -a "$REPORT"
  say "high: loss at steps 1, 10, 30, last; step time:"
  tr '\r' '\n' < "$LOGS/high_train.log" | grep -oE "'loss': [0-9.e+-]+" | awk '{a[NR]=$2} END {print a[1], a[10], a[30], a[NR], "(" NR " steps)"}' \
    | sed 's/^/  /' | tee -a "$REPORT"
  tr '\r' '\n' < "$LOGS/high_train.log" | grep -oE "[0-9]+\.[0-9]+s/it" | tail -1 | sed 's/^/  last: /' | tee -a "$REPORT"
  if [ "$CHECKPOINTS" = 1 ]; then
    [ -f "$out/config.json" ] || { say "high: no saved weights"; return 1; }
    say "high: saved weights $(du -sh "$out" | cut -f1)"
    for t in $TASKS; do
      ev=$W/high_eval/$t
      [ -d "$SFT_ROOT/$t" ] || continue
      if [ ! -f "$ev/summary.json" ]; then
        (cd "$REPO/third_party/memer" && CUDA_VISIBLE_DEVICES=0 "$HOME/envs/memer/bin/python" scripts/eval_subtask_rollout.py \
          --model-path "$out" --output-dir "$ev" --lerobot-path "$DATA/$t" --camera-layout-config "$LAYOUT" \
          --view-width 192 --view-height 192 --episode-indices "$HIGH_EVAL" --device cuda --dtype bfloat16 \
          --attn-implementation sdpa) > "$LOGS/high_eval_$t.log" 2>&1 \
          || { say "high: eval $t FAILED"; tail -n 20 "$LOGS/high_eval_$t.log" >> "$REPORT"; continue; }
      fi
      python3 - "$ev" "$t" <<'EOF' | tee -a "$REPORT"
import collections, json, pathlib, sys
ev, task = pathlib.Path(sys.argv[1]), sys.argv[2]
m = json.loads((ev / "summary.json").read_text())["metrics"]
rows = [json.loads(line) for line in (ev / "predictions.jsonl").open()]
majority = collections.Counter(r["target_subtask"] for r in rows).most_common(1)[0][1] / max(len(rows), 1)
print(f"  {task:14s} frames {m['total_examples']:5d}  subtask accuracy {m['normalized_accuracy']:.3f} "
      f"(majority-label baseline {majority:.3f})  parse failures {m['parse_failure_rate']:.3f}")
EOF
    done
  fi
  [ "${KEEP:-0}" = 1 ] || rm -rf "$out"
  touch "$W/.high_done"
}

mvla() {
  local rlds=$W/rlds t bs run=fetch6 py=$HOME/envs/memvla/bin/python steps runs=$W/runs
  [ -f "$W/.mvla_done" ] && return 0
  mkdir -p "$rlds" "$W/src"
  ln -sfn "$DATA" "$W/src/memer"   # fetch_rlds reads <FETCH_LEROBOT_ROOT>/memer/<task>
  for t in $TASKS; do
    [ -f "$rlds/.built_$t" ] && continue
    say "mvla: RLDS of $t, episodes 0-$((MVLA_EPISODES - 1)) (log $LOGS/rlds_$t.log)"
    FETCH_LEROBOT_ROOT=$W/src FETCH_RLDS_MAX_EPISODES=$MVLA_EPISODES OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= \
      "$py" "$REPO/vla/memoryvla/rlds/build_rlds.py" "fetch_$t" --data-dir "$rlds" > "$LOGS/rlds_$t.log" 2>&1 \
      || { say "mvla: RLDS $t FAILED"; tail -n 20 "$LOGS/rlds_$t.log" >> "$REPORT"; return 1; }
    touch "$rlds/.built_$t"
  done
  say "mvla: RLDS $(du -sh "$rlds" | cut -f1)"
  steps=$MVLA_STEPS; [ "$CHECKPOINTS" = 0 ] && steps=$ENDLESS
  local fit=""
  for bs in $MVLA_BS; do
    rm -rf "$runs/$run--image_aug"
    say "mvla: train.sh on 4 GPUs, per-device batch $bs, $MVLA_STEPS steps (log $runs/logs/$run.log)"
    run_until "mvla_bs$bs" "$MVLA_STEPS" "grep -c 'VLA Train/Loss' '$runs/$run--image_aug/$run--image_aug.jsonl'" -- \
      env BS="$bs" "$REPO/vla/memoryvla/train.sh" "$rlds" "$runs" "$run" "$steps" "$steps"
    local rc=$?
    if grep -qiE "out of memory|OutOfMemoryError" "$runs/logs/$run.log" 2>/dev/null; then
      say "mvla: out of GPU memory at per-device batch $bs after $(grep -c 'VLA Train/Loss' "$runs/$run--image_aug/$run--image_aug.jsonl" 2>/dev/null || echo 0) steps"
      continue
    fi
    [ $rc -eq 0 ] || { say "mvla: training FAILED (exit $rc)"; tail -n 30 "$runs/logs/$run.log" >> "$REPORT"; return 1; }
    fit=$bs; break
  done
  [ -n "$fit" ] || { say "mvla: no per-device batch of '$MVLA_BS' fits"; return 1; }
  "$py" - "$runs/$run--image_aug/$run--image_aug.jsonl" "$bs" <<'EOF' | tee -a "$REPORT"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if '"VLA Train/Loss"' in l]
loss = [r["VLA Train/Loss"] for r in rows]; dt = sorted(r["VLA Train/Step Time"] for r in rows)
mean = lambda x: sum(x) / max(len(x), 1)
print(f"  per-device batch {sys.argv[2]}: loss first 20 steps {mean(loss[:20]):.4f}, last 20 {mean(loss[-20:]):.4f} "
      f"({len(loss)} steps); median step {dt[len(dt) // 2]:.2f} s" if loss else "  loss: not found")
EOF
  if [ "$CHECKPOINTS" = 1 ]; then
    say "mvla: checkpoint $(ls -sh "$runs/$run--image_aug/checkpoints/"*.pt | tail -1)"
    for t in $TASKS; do
      [ -f "$W/mvla_check_$t.json" ] && continue
      CUDA_VISIBLE_DEVICES=0 "$py" "$REPO/vla/memoryvla/check_action_error.py" "$runs/$run--image_aug" "$DATA/$t" \
        "fetch_$t" --episodes "$MVLA_HELD_OUT" --every 8 --out "$W/mvla_check_$t.json" > "$LOGS/mvla_check_$t.log" 2>&1 \
        || { say "mvla: check $t FAILED"; tail -n 20 "$LOGS/mvla_check_$t.log" >> "$REPORT"; continue; }
      python3 - "$W/mvla_check_$t.json" "$t" <<'EOF' | tee -a "$REPORT"
import json, sys
r = json.load(open(sys.argv[1]))
arm = r["arm"].get("moving", {})
print(f"  {sys.argv[2]:14s} calls {r['calls']:4d}  arm model/hold {arm.get('ratio', float('nan')):.2f} (moving chunks)  "
      f"forward |err| {r['forward']['moving_mean_abs_error']}  turn |err| {r['turn']['moving_mean_abs_error']}  "
      f"gripper sign agreement {r['gripper_agreement']:.2f}")
EOF
    done
    [ "${KEEP:-0}" = 1 ] || { rm -rf "$runs/$run--image_aug/checkpoints"; say "mvla: checkpoint deleted"; }
  fi
  touch "$W/.mvla_done"
}

say "four-GPU debug on $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | xargs); repo $(git -C "$REPO" rev-parse --short HEAD) + local changes; stages: $STAGES; checkpoints: $CHECKPOINTS"
for stage in $STAGES; do "$stage" || say "$stage: stage failed, continuing"; done
say "disk: $(du -s --block-size=1M "$W" 2>/dev/null | cut -f1) MB in $W"
say "finished"
