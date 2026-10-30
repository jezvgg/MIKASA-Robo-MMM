#!/usr/bin/env bash
# Detached CabinetSearch AG driver, profile 18 (~/sd-collect/repo-cab18, 3eef80c: the hand collides with the leaves and
# the can, the home rug): batches of 20, JOBS jobs, target TARGET accepted (1020: D6 margin), seeds from 7890000, prefix ag18-.
# MAX_BATCHES=3 runs the pilot. BUDGET counts all of ~/sd-collect/runs-ag. Renders (campaign RGB + the gate's views) share
# GPU_LANES lanes and start with GPU_MIN_FREE MiB free (repo-cab18 4e3ba84; a training job holds ~78 of 80 GB).
source $HOME/sd-collect/env.sh
A=$HOME/sd-collect/driver; mkdir -p $A/pids $A/batches $HOME/sd-collect/runs-ag
if [ -f $A/pids/cabinet_search.pid ] && kill -0 "$(cat $A/pids/cabinet_search.pid)" 2>/dev/null; then echo "already running"; exit 0; fi
cd $A
MIKASA_GPU_LOCK_DIR=$HOME/sd-collect/locks/gpu MIKASA_GPU_LANES=${GPU_LANES:-1} MIKASA_GPU_MIN_FREE_MIB=${GPU_MIN_FREE:-2600} \
  MAX_BATCHES=${MAX_BATCHES:-0} setsid nohup $SIM $A/collection_driver.py cabinet_search 20 ${JOBS:-4} ${TARGET:-1020} ${BUDGET:-80} ag18- >> $A/driver-cabinet_search.out 2>&1 < /dev/null &
echo $! > $A/pids/cabinet_search.pid; echo "started pid $(cat $A/pids/cabinet_search.pid)"
