#!/usr/bin/env bash
# After the CabinetSearch AG driver reached its target: validation100 under profile 16 (candidates 7845000-7845179),
# exact-1000 selection, linked merge, H4 + README (MIT, assisted grasp), checklist, pi0.5 readiness.
set -u
source $HOME/sd-collect/env.sh
D=$HOME/sd-collect/driver; F=$D/final; R=$HOME/sd-collect/runs-ag; OUT=$R/final/cabinetsearch-ag-1000ep
TOK=$HOME/mikasa-pi05/openpi-cache/big_vision/paligemma_tokenizer.model
cd $HOME/sd-collect/repo-cab
step() { echo "== $1 $(date -u +%FT%TZ)"; }
step validation-campaign
[ -f $R/cabinet_search-validation-ag-001/run.json ] && [ -f $F/cabinet_search-ag-validation-100.json ] || \
PYTHONPATH=. $SIM -m utils.collection.campaign --profile cabinet_search --tokenizer $TOK --output $R/cabinet_search-validation-ag-001 \
  --start-seed 7845000 --num-seeds 180 --purpose validation --jobs 8 --through oracle --timeout 1800 || exit 1
step validation-freeze
[ -f $F/cabinet_search-ag-validation-100.json ] || PYTHONPATH=. $SIM -m utils.collection.validation --runs $R/cabinet_search-validation-ag-001 \
  --exclude-runs $R/cabinet_search-ag-train-* --output $F/cabinet_search-ag-validation-100.json || exit 1
step selection
python3 $F/build_selection_cabinet.py $F/cabinet_search-ag-selection.json || exit 1
step merge
inputs=$(python3 -c "import json;print(\" \".join(json.load(open(\"$F/cabinet_search-ag-selection.json\"))[\"inputs\"]))")
HF_HUB_OFFLINE=1 PYTHONPATH=. $EXP -m utils.collection.merge_lerobot --inputs $inputs --output $OUT \
  --repo-id nurtayev-d/cabinetsearch-ag-1000ep --link-videos --select $F/cabinet_search-ag-selection.json || exit 1
step finalize
python3 $F/finalize_dataset.py $OUT $F/cabinet_search-ag-validation-100.json "MIKASA CabinetSearch (DSFetch, assisted grasp)" || exit 1
step checklist
HF_HUB_OFFLINE=1 PYTHONPATH=. $EXP -m utils.collection.checklist --dataset $OUT --output $F/cabinetsearch-ag-1000ep-checklist.json \
  --load-samples --tokenizer $TOK > $F/cabinetsearch-ag-1000ep-checklist.log 2>&1; echo "checklist exit=$?"
step vla-ready
HF_HUB_OFFLINE=1 $EXP $F/vla_ready.py $F/vla-cabinetsearch-ag-1000ep.json $OUT > $F/vla-cabinetsearch-ag-1000ep.log 2>&1; echo "vla exit=$?"
# no rejected-RGB prune: owner 2026-10-19 «теперь можно не удалять» (+300 GB); the sources are on HF
du -sh $OUT; step done; date > $F/FINALIZE_CABINET_DONE
