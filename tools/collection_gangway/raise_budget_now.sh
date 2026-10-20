#!/usr/bin/env bash
# Restart the SameDrawer AG driver with a larger disk budget between campaigns (its gates keep running and
# write their ledgers; the relaunched driver skips gates that still hold their lock).
A=$HOME/sd-collect/driver
camp() { ps -eo args | grep "[u]tils.collection.campaign" | grep sd-collect | grep -q "profile same_drawer"; }
while camp; do sleep 15; done
P=$(cat $A/pids/same_drawer.pid); C=$(tr "\0" " " < /proc/$P/cmdline 2>/dev/null)
case "$C" in *sd-collect/driver/collection_driver.py*same_drawer*) kill $P; echo "stopped driver $P at $(date -u)";; *) echo "pid $P is not the driver: $C"; exit 1;; esac
sleep 5; while camp; do echo "a campaign started before the stop; waiting"; sleep 30; done
rm -f $A/pids/same_drawer.pid
BUDGET=${NEW_BUDGET:-42} bash $HOME/sd-collect/launch.sh
