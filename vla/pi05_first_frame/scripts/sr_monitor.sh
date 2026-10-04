#!/usr/bin/env bash
# Progress SR while training: evaluate each new params-only snapshot on the 20 dev seeds
# (dev_seeds.json: planner-solved, in neither the training episodes nor the 100 validation seeds).
#
#   sr_monitor.sh CONFIG EXP [GPU] [EPISODES]        (source server.env first; run under nohup)
#
# Snapshots come from `python -m mikasa_pi05 train` ($PI05_WORK/snapshots/CONFIG/EXP/<step>, every
# save_interval steps). The newest one is evaluated each time; if evaluation falls behind, older
# snapshots are skipped rather than queued. One line per evaluated step goes to
# $PI05_WORK/logs/sr-EXP.txt (and machine-readable to sr-EXP.tsv). Exits after the last training
# step is evaluated, or when training has stopped and no new snapshot appeared for an hour.
# Next to training, the policy server takes only SERVER_MEM_FRACTION of the GPU (default 0.12, ~10 GB
# of 80); start training with XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 so that room exists.
set -uo pipefail
: "${OPENPI_DIR:?source server.env first}" "${SIM_VENV:?}" "${REPO_DIR:?}" "${PI05_WORK:?}"
CONFIG="$1"; EXP="$2"; GPU="${3:-3}"; EPISODES="${4:-20}"
read -r -a EXTRA <<< "${SR_EVAL_ARGS:-}"   # extra eval_samedrawer.py flags (tests: --max-policy-steps 20)
HERE="$REPO_DIR/vla/pi05_first_frame"
SNAPSHOTS="$PI05_WORK/snapshots/$CONFIG/$EXP"
OUT="$PI05_WORK/results/sr-$EXP"
TXT="$PI05_WORK/logs/sr-$EXP.txt"
TSV="$PI05_WORK/logs/sr-$EXP.tsv"
export SERVER_MEM_FRACTION="${SERVER_MEM_FRACTION:-0.12}" PORT_BASE="${PORT_BASE:-8200}"
LAST=$("$OPENPI_DIR/.venv/bin/python" -c "from mikasa_pi05.configs import get_config; print(get_config('$CONFIG').num_train_steps - 1)" 2>/dev/null)
mkdir -p "$OUT" "$PI05_WORK/logs"
[[ -f "$TSV" ]] || printf 'time\tstep\tsuccess\tepisodes\tsr\twilson_lo\twilson_hi\tclosed_cue\tapple_placed\twrong_drawer\n' > "$TSV"
say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$TXT"; }
say "monitor $CONFIG/$EXP on GPU $GPU, $EPISODES dev seeds per snapshot, last step $LAST"

idle_since=$(date +%s)
while true; do
  step=$(ls -1 "$SNAPSHOTS" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  if [[ -n "$step" && ! -f "$OUT/$step/summary.json" && ! -f "$OUT/$step/failed" ]]; then
    say "step $step: evaluating"
    "$HERE/scripts/eval_parallel.sh" "$CONFIG" "$SNAPSHOTS/$step" "$OUT/$step" "$GPU" \
        --seeds "$HERE/dev_seeds.json" --max-episodes "$EPISODES" "${EXTRA[@]}" > "$OUT/eval-$step.log" 2>&1
    if [[ -f "$OUT/$step/summary.json" ]]; then
      "$SIM_VENV/bin/python" - "$OUT/$step/summary.json" "$step" "$TSV" <<'PY' | tee -a "$TXT"
import json, sys, time
s = json.load(open(sys.argv[1])); step = sys.argv[2]; st = s["stages"]
lo, hi = s["wilson95"]
row = [time.strftime("%F %T"), step, s["success"], s["episodes"], round(s["success_rate"], 3), lo, hi,
       st["closed_cue_drawer"], st["apple_placed"], st["wrong_drawer_touched"]]
open(sys.argv[3], "a").write("\t".join(map(str, row)) + "\n")
print(f"  step {step}: SR {s['success']}/{s['episodes']} = {s['success_rate']:.2f} (95% CI {lo:.2f}-{hi:.2f}); "
      f"closed cue {st['closed_cue_drawer']}, apple placed {st['apple_placed']}, wrong drawer {st['wrong_drawer_touched']}")
PY
    else
      say "step $step: evaluation produced no summary (see $OUT/eval-$step.log)"
      mkdir -p "$OUT/$step" && echo '{}' > "$OUT/$step/failed"
    fi
    idle_since=$(date +%s)
    [[ "$step" == "$LAST" ]] && { say "last step evaluated; monitor done"; exit 0; }
    continue
  fi
  if ! pgrep -f "mikasa_pi05 train $CONFIG" >/dev/null && (( $(date +%s) - idle_since > 3600 )); then
    say "training not running and no new snapshot for an hour; monitor done"; exit 0
  fi
  sleep 60
done
