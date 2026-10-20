#!/usr/bin/env bash
# Detached SameDrawer collection driver (second 1000): batches of 20, 8 jobs (campaign allows 1-8), target 1048 accepted (profile 21, assisted grasp),
# pause above BUDGET GB in ~/sd-collect/runs-ag, batch prefix ag-. PID in ~/sd-collect/driver/pids/same_drawer.pid
source $HOME/sd-collect/env.sh
A=$HOME/sd-collect/driver; mkdir -p $A/pids $A/batches $HOME/sd-collect/runs-ag
if [ -f $A/pids/same_drawer.pid ] && kill -0 "$(cat $A/pids/same_drawer.pid)" 2>/dev/null; then echo "already running"; exit 0; fi
cd $A
setsid nohup $SIM $A/collection_driver.py same_drawer 20 8 ${TARGET:-1048} ${BUDGET:-10} ag- >> $A/driver-same_drawer.out 2>&1 < /dev/null &
echo $! > $A/pids/same_drawer.pid; echo "started pid $(cat $A/pids/same_drawer.pid)"
