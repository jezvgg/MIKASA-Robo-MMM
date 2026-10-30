#!/usr/bin/env bash
# Owner 2026-10-26 «Если прямо сейчас запустить 10 сидов новых»: SeasonDish SR on 10 fresh development seeds
# 7792000-7792009 (never used; SEED_PLAN amendment), through the held 10 Hz replay (validated), twice:
# the collection's pinned code (repo-sdish = 5dbf5a8, profile 26) and the current train/vla (85953b3).
source $HOME/sd-collect/env.sh
TOK=$HOME/mikasa-pi05/openpi-cache/big_vision/paligemma_tokenizer.model
O=$HOME/sd-collect/sdish-check; mkdir -p $O
for pair in "p26-5dbf5a8:$HOME/sd-collect/repo-sdish" "trainvla-85953b3:$HOME/train-vla-check2/repo"; do
  name=${pair%%:*}; repo=${pair#*:}
  echo "== $name $(git -C $repo log --oneline -1) $(date -u +%FT%TZ)"
  (cd $repo && PYTHONPATH=. $SIM -m utils.collection.campaign --profile season_dish --tokenizer $TOK --output $O/$name \
     --start-seed 7792000 --num-seeds 10 --purpose development --jobs 8 --through validated --timeout 1800) \
     > $O/$name.log 2>&1
  echo "campaign exit $? $(date -u +%FT%TZ)"
done
echo "ALL DONE $(date -u +%FT%TZ)"
