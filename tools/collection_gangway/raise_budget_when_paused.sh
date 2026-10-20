#!/usr/bin/env bash
# Wait for the SameDrawer AG driver to report PAUSED (no PAUSED line exists yet), make sure no campaign runs,
# stop ONLY the driver process (its postbatch gates keep running and write their ledgers), relaunch with a larger budget.
A=$HOME/sd-collect/driver
until grep -q "PAUSED" $A/driver-same_drawer.out; do sleep 30; done
while ps -eo args | grep "[u]tils.collection.campaign" | grep -q sd-collect; do sleep 30; done
P=$(cat $A/pids/same_drawer.pid); C=$(tr "\0" " " < /proc/$P/cmdline 2>/dev/null)
case "$C" in *sd-collect/driver/collection_driver.py*) kill $P; echo "stopped driver $P at $(date -u)";; *) echo "pid $P is not the driver: $C"; exit 1;; esac
sleep 5; rm -f $A/pids/same_drawer.pid
BUDGET=${NEW_BUDGET:-28} bash $HOME/sd-collect/launch.sh
