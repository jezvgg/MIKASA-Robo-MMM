#!/usr/bin/env bash
# CANCELLED 2026-10-25 (owner: «Теперь задача на 4 отсеках не нужна. Фиксируем на 5 отсеках»): do not relaunch; see SEED_PLAN.
echo "cancelled: the four-compartment collection is not resumed (SEED_PLAN, 2026-10-25)"; exit 1
# Detached CabinetSearch AG driver, profile 19 (~/sd-collect/repo-cab19 = d674243 on check/assisted-grasp-collisions:
# the hand collides with the leaves and the can, the home rug, the touch readied 10 cm short of the dock): batches
# of 20, JOBS jobs (owner 2026-10-24: 6), target TARGET accepted (1020: D6 margin), seeds from 7890000, prefix ag19-.
# MAX_BATCHES=3 runs the pilot. BUDGET counts all of ~/sd-collect/runs-ag. The stock campaign (no render lanes).
source $HOME/sd-collect/env.sh
A=$HOME/sd-collect/driver; mkdir -p $A/pids $A/batches $HOME/sd-collect/runs-ag
if [ -f $A/pids/cabinet_search.pid ] && kill -0 "$(cat $A/pids/cabinet_search.pid)" 2>/dev/null; then echo "already running"; exit 0; fi
cd $A
MAX_BATCHES=${MAX_BATCHES:-0} setsid nohup $SIM $A/collection_driver.py cabinet_search 20 ${JOBS:-6} ${TARGET:-1020} ${BUDGET:-80} ag19- >> $A/driver-cabinet_search.out 2>&1 < /dev/null &
echo $! > $A/pids/cabinet_search.pid; echo "started pid $(cat $A/pids/cabinet_search.pid)"
