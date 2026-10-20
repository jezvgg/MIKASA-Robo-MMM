#!/usr/bin/env bash
# Detached DepthRecall AG collection driver: profile 5 (~/sd-collect/repo-dr, ed81c42), batches of 10 (RGB h5 0.5 GB/ep), 8 jobs,
# target 1000 accepted, seeds from 7970000, prefix ag-. BUDGET counts all of ~/sd-collect/runs-ag (SameDrawer AG included).
source $HOME/sd-collect/env.sh
A=$HOME/sd-collect/driver; mkdir -p $A/pids $A/batches $HOME/sd-collect/runs-ag
if [ -f $A/pids/depth_recall.pid ] && kill -0 "$(cat $A/pids/depth_recall.pid)" 2>/dev/null; then echo "already running"; exit 0; fi
cd $A
setsid nohup $SIM $A/collection_driver.py depth_recall 10 8 ${TARGET:-1000} ${BUDGET:-45} ag- >> $A/driver-depth_recall.out 2>&1 < /dev/null &
echo $! > $A/pids/depth_recall.pid; echo "started pid $(cat $A/pids/depth_recall.pid)"
