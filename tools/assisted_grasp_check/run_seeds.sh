#!/usr/bin/env bash
# Run episode_check.py for CabinetSearch on a list of seeds, at most four at a time, then summarize.
# usage: run_seeds.sh <out dir> <seed>...   (extra settings through the environment, e.g.
#        MIKASA_BD_HAND_TURN=0.25 run_seeds.sh ...; the sim interpreter is $SIM_VENV/bin/python)
set -u
OUT=$1; shift
REPO=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"; cd "$REPO"
lane=0
for s in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 20; done
  PYTHONPATH=. taskset -c $lane "$SIM_VENV/bin/python" -W ignore tools/assisted_grasp_check/episode_check.py \
    cabinet_search "$s" --assisted-grasp > "$OUT/$s.log" 2>&1 &
  lane=$(( (lane + 1) % 4 ))
done
wait
for s in "$@"; do
  f=$OUT/$s.log
  v=$(grep -h -o -E "verdict=[a-z_]+" "$f" | head -1)
  c=$(grep -h -o -E "passed=(True|False)" "$f" | tail -1)
  x=$(grep -h -o -E "max_xy_m=[0-9.]+ max_tilt_deg=[0-9.]+" "$f" | tail -1)
  h=$(grep -h -o -E "panel_depth_max_mm=[0-9.]+|press_max_n=[0-9.]+|steps_press_gt20n=[0-9]+" "$f" | tr "\n" " ")
  sw=$(grep -h -o -E "swing done \(bimbo\) rad=[0-9.]+ door_from_hand=[0-9.]+ why=[a-z-]+ steps=[0-9]+" "$f" \
       | sed -E "s/swing done \(bimbo\) //; s/door_from_hand=[0-9.]+ //" | tr "\n" ";")
  echo "$s | $v | can_early $c $x | $h| tracebacks=$(grep -c Traceback "$f") | $sw"
done > "$OUT/SUMMARY.txt"
echo ALL_DONE >> "$OUT/SUMMARY.txt"
