#!/usr/bin/env bash
# After the CabinetSearch5 AG driver reached its target: validation100 under profile 7 (candidates 7847000-7847179),
# exact-1000 selection, linked merge, H4 + README (MIT, assisted grasp, five-compartment notes), checklist, pi0.5 readiness.
set -u
source $HOME/sd-collect/env.sh
D=$HOME/sd-collect/driver; F=$D/final; R=$HOME/sd-collect/runs-ag; OUT=$R/final/cabinetsearch5-ag-1000ep
TOK=$HOME/mikasa-pi05/openpi-cache/big_vision/paligemma_tokenizer.model
cd $HOME/sd-collect/repo-cab5c7
step() { echo "== $1 $(date -u +%FT%TZ)"; }
step validation-campaign
[ -f $R/cabinet_search5-validation-c5v7-001/run.json ] && [ -f $F/cabinet_search5-c5v7-validation-100.json ] || \
PYTHONPATH=. $SIM -m utils.collection.campaign --profile cabinet_search5 --tokenizer $TOK --output $R/cabinet_search5-validation-c5v7-001 \
  --start-seed 7847000 --num-seeds 180 --purpose validation --jobs 8 --through oracle --timeout 1800 || exit 1
step validation-freeze
[ -f $F/cabinet_search5-c5v7-validation-100.json ] || PYTHONPATH=. $SIM -m utils.collection.validation --runs $R/cabinet_search5-validation-c5v7-001 \
  --exclude-runs $R/cabinet_search5-c5v7-train-* --output $F/cabinet_search5-c5v7-validation-100.json || exit 1
step selection
$SIM $F/build_selection_cabinet5.py $F/cabinet_search5-c5v7-selection.json || exit 1
step merge
inputs=$(python3 -c "import json;print(\" \".join(json.load(open(\"$F/cabinet_search5-c5v7-selection.json\"))[\"inputs\"]))")
HF_HUB_OFFLINE=1 PYTHONPATH=. $EXP -m utils.collection.merge_lerobot --inputs $inputs --output $OUT \
  --repo-id nurtayev-d/cabinetsearch5-ag-1000ep --link-videos --select $F/cabinet_search5-c5v7-selection.json || exit 1
step finalize
python3 $F/finalize_dataset.py $OUT $F/cabinet_search5-c5v7-validation-100.json "MIKASA CabinetSearch5 (DSFetch, assisted grasp, five compartments)" || exit 1
python3 $F/readme_cabinet5_notes.py $OUT $F/cabinet_search5-c5v7-selection.json || exit 1
step checklist
HF_HUB_OFFLINE=1 PYTHONPATH=. $EXP -m utils.collection.checklist --dataset $OUT --output $F/cabinetsearch5-ag-1000ep-checklist.json \
  --load-samples --tokenizer $TOK > $F/cabinetsearch5-ag-1000ep-checklist.log 2>&1; echo "checklist exit=$?"
step vla-ready
HF_HUB_OFFLINE=1 $EXP $F/vla_ready.py $F/vla-cabinetsearch5-ag-1000ep.json $OUT > $F/vla-cabinetsearch5-ag-1000ep.log 2>&1; echo "vla exit=$?"
du -sh $OUT; step done; date > $F/FINALIZE_CABINET5_DONE
