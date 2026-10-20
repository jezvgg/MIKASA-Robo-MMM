#!/usr/bin/env bash
# Detached SeasonDish AG collection driver: profile 26 (~/sd-collect/repo-sdish, 5dbf5a8), batches of 20, 8 jobs,
# target 1000 accepted, seeds from 7780000, prefix ag-. BUDGET counts all of ~/sd-collect/runs-ag (SameDrawer AG included).
source $HOME/sd-collect/env.sh
A=$HOME/sd-collect/driver; mkdir -p $A/pids $A/batches $HOME/sd-collect/runs-ag
if [ -f $A/pids/season_dish.pid ] && kill -0 "$(cat $A/pids/season_dish.pid)" 2>/dev/null; then echo "already running"; exit 0; fi
cd $A
setsid nohup $SIM $A/collection_driver.py season_dish 20 8 ${TARGET:-1000} ${BUDGET:-45} ag- >> $A/driver-season_dish.out 2>&1 < /dev/null &
echo $! > $A/pids/season_dish.pid; echo "started pid $(cat $A/pids/season_dish.pid)"
