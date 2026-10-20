#!/usr/bin/env bash
# After the SeasonDish AG driver reached its target: validation100 under profile 26 (candidates 7745000-7745179),
# exact-1000 selection, linked merge, H4 + README (MIT, assisted grasp), checklist, pi0.5 readiness.
set -u
source $HOME/sd-collect/env.sh
D=$HOME/sd-collect/driver; F=$D/final; R=$HOME/sd-collect/runs-ag; OUT=$R/final/seasondish-ag-1000ep
TOK=$HOME/mikasa-pi05/openpi-cache/big_vision/paligemma_tokenizer.model
cd $HOME/sd-collect/repo-sdish
step() { echo "== $1 $(date -u +%FT%TZ)"; }
step validation-campaign
[ -f $R/season_dish-validation-ag-001/run.json ] && [ -f $F/season_dish-ag-validation-100.json ] || \
PYTHONPATH=. $SIM -m utils.collection.campaign --profile season_dish --tokenizer $TOK --output $R/season_dish-validation-ag-001 \
  --start-seed 7745000 --num-seeds 180 --purpose validation --jobs 8 --through oracle --timeout 1800 || exit 1
step validation-freeze
[ -f $F/season_dish-ag-validation-100.json ] || PYTHONPATH=. $SIM -m utils.collection.validation --runs $R/season_dish-validation-ag-001 \
  --exclude-runs $R/season_dish-ag-train-* --output $F/season_dish-ag-validation-100.json || exit 1
step selection
python3 $F/build_selection_season.py $F/season_dish-ag-selection.json || exit 1
step merge
inputs=$(python3 -c "import json;print(\" \".join(json.load(open(\"$F/season_dish-ag-selection.json\"))[\"inputs\"]))")
HF_HUB_OFFLINE=1 PYTHONPATH=. $EXP -m utils.collection.merge_lerobot --inputs $inputs --output $OUT \
  --repo-id nurtayev-d/seasondish-ag-1000ep --link-videos --select $F/season_dish-ag-selection.json || exit 1
step finalize
python3 $F/finalize_dataset.py $OUT $F/season_dish-ag-validation-100.json "MIKASA SeasonDish (DSFetch, assisted grasp)" || exit 1
step checklist
HF_HUB_OFFLINE=1 PYTHONPATH=. $EXP -m utils.collection.checklist --dataset $OUT --output $F/seasondish-ag-1000ep-checklist.json \
  --load-samples --tokenizer $TOK > $F/seasondish-ag-1000ep-checklist.log 2>&1; echo "checklist exit=$?"
step vla-ready
HF_HUB_OFFLINE=1 $EXP $F/vla_ready.py $F/vla-seasondish-ag-1000ep.json $OUT > $F/vla-seasondish-ag-1000ep.log 2>&1; echo "vla exit=$?"
step prune-rejected-rgb
python3 $F/prune_rejected_rgb.py season_dish
du -sh $OUT; step done; date > $F/FINALIZE_SEASONDISH_DONE
