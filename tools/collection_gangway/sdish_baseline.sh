#!/usr/bin/env bash
# SeasonDish planner baseline (owner 2026-10-27, before the planner change): the planner's SR and failure reasons on a
# fixed development seed set, rerun unchanged after the change and compared seed by seed (sdish_tally.py --compare).
# The seeds are declared in SEED_PLAN before the first run (7793000-7793199); the same seeds for every arm.
#
#   sdish_baseline.sh REPO NAME [THROUGH] [JOBS] [START] [N]
#     REPO     a fresh clone at the commit under test (never one a campaign runs from; the run pins its signature)
#     NAME     the run dir under ~/sd-collect/sdplan
#     THROUGH  oracle (default: the planner verdict only) or validated (also the native and held 10 Hz replays)
#     JOBS     workers, at most 4 (default 4)
source $HOME/sd-collect/env.sh
TOK=$HOME/mikasa-pi05/openpi-cache/big_vision/paligemma_tokenizer.model
R=$1; NAME=$2; THROUGH=${3:-oracle}; JOBS=${4:-4}; START=${5:-7793000}; N=${6:-200}
C=$HOME/sd-collect/sdplan; O=$C/$NAME
[ -d "$R" ] && [ -n "$NAME" ] || { echo "usage: $0 REPO NAME [THROUGH] [JOBS] [START] [N]"; exit 2; }
[ "$JOBS" -le 4 ] || { echo "at most 4 workers"; exit 2; }
mkdir -p $C; cd $R
echo "== $NAME: $(git log --oneline -1) seeds $START+$N through $THROUGH, $JOBS workers, $(date -u +%FT%TZ)"
PYTHONPATH=. $SIM -m utils.collection.campaign --profile season_dish --tokenizer $TOK --output $O --start-seed $START \
  --num-seeds $N --purpose development --jobs $JOBS --through $THROUGH --timeout 1800 > $O.log 2>&1
echo "campaign exit $? $(date -u +%FT%TZ)"
python3 $R/tools/collection_gangway/checks/sdish_tally.py --json $O.tally.json $O
