#!/usr/bin/env bash
# Detached five-compartment CabinetSearch driver (cabinet_search5 profile 7, ~/sd-collect/repo-cab5c7 = d65cd4c on
# feat/cabinet-fifth-door, unpushed): batches of 20, JOBS jobs (default 8), target TARGET accepted (1020: D6 margin),
# seeds from 7920700, prefix c5v7-. BUDGET counts all of ~/sd-collect/runs-ag (default 110 GB here: the paused
# profile-19 runs hold ~35 GB). Owner 2026-10-24: «приоритет дофиксить версию с 5 отсеками и записать ее».
source $HOME/sd-collect/env.sh
A=$HOME/sd-collect/driver; mkdir -p $A/pids $A/batches $HOME/sd-collect/runs-ag
if [ -f $A/pids/cabinet_search5.pid ] && kill -0 "$(cat $A/pids/cabinet_search5.pid)" 2>/dev/null; then echo "already running"; exit 0; fi
cd $A
MAX_BATCHES=${MAX_BATCHES:-0} setsid nohup $SIM $A/collection_driver.py cabinet_search5 20 ${JOBS:-8} ${TARGET:-1020} ${BUDGET:-110} c5v7- 7920700 >> $A/driver-cabinet_search5.out 2>&1 < /dev/null &
echo $! > $A/pids/cabinet_search5.pid; echo "started pid $(cat $A/pids/cabinet_search5.pid)"
