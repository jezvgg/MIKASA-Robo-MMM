#!/usr/bin/env bash
# Owner 2026-10-26: SeasonDish on 100 fresh development seeds 7792010-7792109 (SEED_PLAN), train/vla 85953b3, through the
# held 10 Hz replay, then the interpolated second step (interp_replay.py all / nogrip).
source $HOME/sd-collect/env.sh
TOK=$HOME/mikasa-pi05/openpi-cache/big_vision/paligemma_tokenizer.model
C=$HOME/sd-collect/sdish-check; R=$HOME/train-vla-check2/repo; O=$C/trainvla-100
cd $R
echo "== campaign $(git log --oneline -1) $(date -u +%FT%TZ)"
PYTHONPATH=. $SIM -m utils.collection.campaign --profile season_dish --tokenizer $TOK --output $O --start-seed 7792010 --num-seeds 100 \
  --purpose development --jobs 8 --through validated --timeout 1800 > $O.log 2>&1; echo "campaign exit $? $(date -u +%FT%TZ)"
for m in all nogrip; do PYTHONPATH=. $SIM $C/interp_replay.py $O $m 8 2>&1 | grep -v -i -E "vulkan|warn\(|Mimic targets" | tail -n 1; done
echo "ALL DONE $(date -u +%FT%TZ)"
