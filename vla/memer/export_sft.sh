#!/usr/bin/env bash
# MemER high level's SFT data for the full run: the authors' generate_sft_data.py on every episode
# of the six labelled copies ($MEMER_DATA_ROOT/<task>, label_subtasks.py), with the settings the
# debug run used (192-px views, frame_subsample 5, 8 recent frames, 8 keyframes, horizon 2).
# One task at a time, $WORKERS processes (default 4). A task with a train.json is skipped.
# Entries are named episode_NNNNNN_tTTTTTT, so held-out episodes can be filtered out at launch.
# usage: nohup vla/memer/export_sft.sh [out dir, default ~/datasets/memer_sft] > log 2>&1 < /dev/null &
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$HOME/datasets/memer_sft}
TASKS=${TASKS:-samedrawer seasondish cabinetsearch depthrecall takeitback veggie}
DATA=${MEMER_DATA_ROOT:-$HOME/datasets/memer}
LAYOUT=$REPO/vla/memer/configs/fetch_stacked_rgb_camera_layout.json
WORKERS=${WORKERS:-4}
mkdir -p "$OUT/logs"
for t in $TASKS; do
  [ -f "$OUT/$t/train.json" ] && { echo "$t: done before"; continue; }
  rm -rf "${OUT:?}/$t"
  echo "$(date '+%F %T') $t: start"
  (cd "$REPO/third_party/memer" && OMP_NUM_THREADS=1 "$HOME/envs/memer/bin/python" scripts/generate_sft_data.py \
    --lerobot_path "$DATA/$t" --output_dir "$OUT/$t" --camera_layout_config "$LAYOUT" \
    --keyframe_rule_file "$DATA/$t/keyframe_rules.json" --frame_subsample 5 --recent_frames_length 8 \
    --keyframes_length 8 --prediction_horizon 2 --view_width 192 --view_height 192 \
    --num_workers "$WORKERS") > "$OUT/logs/$t.log" 2>&1
  echo "$(date '+%F %T') $t: rc=$? $(tail -n 1 "$OUT/logs/$t.log" | cut -c1-200)"
done
echo DONE
