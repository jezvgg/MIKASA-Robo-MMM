#!/usr/bin/env bash
# Owner 2026-10-26 «Да» to the one-branch plan, step 3: the hand inside the graspable objects, 100 episodes per task
# (0, 10, ..., 990) of the published SameDrawer, SeasonDish and DepthRecall AG datasets, replayed with
# repro/one-branch 3d89bae (their default now: the direct drive they were recorded with). The datasets are read
# from ~/datasets (read-only). 8 chunks of ~13 episodes per task, 8 at a time.
source $HOME/sd-collect/env.sh
export R=$HOME/sd-collect/repo-vla-repro O=$HOME/sd-collect/repro-check/overlap
one() {
  local d=$1 c=$2
  local eps
  eps=$(python3 -c "print(' '.join(str(e) for e in range(0, 1000, 10)[$c::8]))")
  (cd $R && PYTHONPATH=. $SIM tools/assisted_grasp_check/hand_overlap_replay.py $HOME/datasets/$d $O/$d-$c.json $eps > $O/$d-$c.log 2>&1)
  echo "$d chunk $c exit $? $(date -u +%T)"
}
export -f one
for d in samedrawer-ag-1000ep seasondish-ag-1000ep depthrecall-ag-1000ep; do
  for c in 0 1 2 3 4 5 6 7; do echo "$d $c"; done
done | xargs -P 8 -n 2 bash -c 'one "$0" "$1"'
echo "ALL DONE $(date -u +%FT%TZ)"
