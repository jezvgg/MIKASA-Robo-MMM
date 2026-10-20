#!/usr/bin/env bash
# After the DepthRecall AG driver reached its target: validation100 under profile 6 (candidates 7965000-7965179),
# exact-1000 selection, linked merge, H4 + README (MIT, assisted grasp), checklist, pi0.5 readiness.
set -u
source $HOME/sd-collect/env.sh
D=$HOME/sd-collect/driver; F=$D/final; R=$HOME/sd-collect/runs-ag; OUT=$R/final/depthrecall-ag-1000ep
TOK=$HOME/mikasa-pi05/openpi-cache/big_vision/paligemma_tokenizer.model
cd $HOME/sd-collect/repo-dr6
step() { echo "== $1 $(date -u +%FT%TZ)"; }
step validation-campaign
[ -f $R/depth_recall-validation-ag6-001/run.json ] && [ -f $F/depth_recall-ag6-validation-100.json ] || \
PYTHONPATH=. $SIM -m utils.collection.campaign --profile depth_recall --tokenizer $TOK --output $R/depth_recall-validation-ag6-001 \
  --start-seed 7965000 --num-seeds 180 --purpose validation --jobs 8 --through oracle --timeout 1800 || exit 1
step validation-freeze
[ -f $F/depth_recall-ag6-validation-100.json ] || PYTHONPATH=. $SIM -m utils.collection.validation --runs $R/depth_recall-validation-ag6-001 \
  --exclude-runs $R/depth_recall-ag6-train-* --output $F/depth_recall-ag6-validation-100.json || exit 1
step selection
python3 $F/build_selection_dr6.py $F/depth_recall-ag6-selection.json || exit 1
step merge
inputs=$(python3 -c "import json;print(\" \".join(json.load(open(\"$F/depth_recall-ag6-selection.json\"))[\"inputs\"]))")
HF_HUB_OFFLINE=1 PYTHONPATH=. $EXP -m utils.collection.merge_lerobot --inputs $inputs --output $OUT \
  --repo-id nurtayev-d/depthrecall-ag-1000ep --link-videos --select $F/depth_recall-ag6-selection.json || exit 1
step finalize
python3 $F/finalize_dataset.py $OUT $F/depth_recall-ag6-validation-100.json "MIKASA DepthRecall (DSFetch, assisted grasp)" || exit 1
step checklist
HF_HUB_OFFLINE=1 PYTHONPATH=. $EXP -m utils.collection.checklist --dataset $OUT --output $F/depthrecall-ag-1000ep-checklist.json \
  --load-samples --tokenizer $TOK > $F/depthrecall-ag-1000ep-checklist.log 2>&1; echo "checklist exit=$?"
step vla-ready
HF_HUB_OFFLINE=1 $EXP $F/vla_ready.py $F/vla-depthrecall-ag-1000ep.json $OUT > $F/vla-depthrecall-ag-1000ep.log 2>&1; echo "vla exit=$?"
# no rejected-RGB prune: owner 2026-10-19 «теперь можно не удалять» (+300 GB); the sources are on HF
du -sh $OUT; step done; date > $F/FINALIZE_DR_DONE
