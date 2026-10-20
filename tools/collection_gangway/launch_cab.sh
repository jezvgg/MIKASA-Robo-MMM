#!/usr/bin/env bash
# Detached CabinetSearch AG collection driver: profile 16 (~/sd-collect/repo-cab, 4562981), batches of 20, 8 jobs,
# target TARGET accepted (1020: D6 margin), seeds from 7880000, prefix ag-. BUDGET counts all of ~/sd-collect/runs-ag (SameDrawer AG included).
source $HOME/sd-collect/env.sh
A=$HOME/sd-collect/driver; mkdir -p $A/pids $A/batches $HOME/sd-collect/runs-ag
if [ -f $A/pids/cabinet_search.pid ] && kill -0 "$(cat $A/pids/cabinet_search.pid)" 2>/dev/null; then echo "already running"; exit 0; fi
cd $A
setsid nohup $SIM $A/collection_driver.py cabinet_search 20 8 ${TARGET:-1020} ${BUDGET:-45} ag- >> $A/driver-cabinet_search.out 2>&1 < /dev/null &
echo $! > $A/pids/cabinet_search.pid; echo "started pid $(cat $A/pids/cabinet_search.pid)"
