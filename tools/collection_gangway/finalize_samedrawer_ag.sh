#!/usr/bin/env bash
# After the SameDrawer AG driver reached its target: validation100 under profile 21, exact-1000 selection,
# linked merge, H4 + README (MIT, assisted grasp), checklist, pi0.5 readiness. Logs in driver/final/finalize.log.
set -u
source $HOME/sd-collect/env.sh
D=$HOME/sd-collect/driver; F=$D/final; R=$HOME/sd-collect/runs-ag; OUT=$R/final/samedrawer-ag-1000ep
TOK=$HOME/mikasa-pi05/openpi-cache/big_vision/paligemma_tokenizer.model
cd $HOME/sd-collect/repo
step() { echo "== $1 $(date -u +%FT%TZ)"; }
step validation-campaign
[ -f $R/same_drawer-validation-ag-001/run.json ] && [ -f $F/same_drawer-ag-validation-100.json ] || \
PYTHONPATH=. $SIM -m utils.collection.campaign --profile same_drawer --tokenizer $TOK --output $R/same_drawer-validation-ag-001 \
  --start-seed 8250000 --num-seeds 140 --purpose validation --jobs 8 --through oracle --timeout 1800 || exit 1
step validation-freeze
[ -f $F/same_drawer-ag-validation-100.json ] || PYTHONPATH=. $SIM -m utils.collection.validation --runs $R/same_drawer-validation-ag-001 \
  --exclude-runs $R/same_drawer-ag-train-* --output $F/same_drawer-ag-validation-100.json || exit 1
step selection
python3 $F/build_selection.py $F/same_drawer-ag-selection.json || exit 1
step merge
inputs=$(python3 -c "import json;print(\" \".join(json.load(open(\"$F/same_drawer-ag-selection.json\"))[\"inputs\"]))")
HF_HUB_OFFLINE=1 PYTHONPATH=. $EXP -m utils.collection.merge_lerobot --inputs $inputs --output $OUT \
  --repo-id nurtayev-d/samedrawer-ag-1000ep --link-videos --select $F/same_drawer-ag-selection.json || exit 1
step finalize
python3 $F/finalize_dataset.py $OUT $F/same_drawer-ag-validation-100.json "MIKASA SameDrawer (DSFetch, assisted grasp)" || exit 1
step checklist
HF_HUB_OFFLINE=1 PYTHONPATH=. $EXP -m utils.collection.checklist --dataset $OUT --output $F/samedrawer-ag-1000ep-checklist.json \
  --load-samples --tokenizer $TOK > $F/samedrawer-ag-1000ep-checklist.log 2>&1; echo "checklist exit=$?"
step vla-ready
HF_HUB_OFFLINE=1 $EXP $F/vla_ready.py $F/vla-samedrawer-ag-1000ep.json $OUT > $F/vla-samedrawer-ag-1000ep.log 2>&1; echo "vla exit=$?"
step prune-rejected-rgb
python3 $F/prune_rejected_rgb.py same_drawer
du -sh $OUT; step done; date > $F/FINALIZE_DONE
