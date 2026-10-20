#!/usr/bin/env bash
# Restart the DepthRecall AG driver between campaigns with BUDGET/TARGET from the environment (gates keep running;
# the relaunched driver skips gates that still hold their lock and resumes a declared, unfinished batch).
A=$HOME/sd-collect/driver
camp() { ps -eo args | grep "[u]tils.collection.campaign" | grep sd-collect | grep -q "profile depth_recall"; }
while camp; do sleep 15; done
P=$(cat $A/pids/depth_recall.pid); C=$(tr "\0" " " < /proc/$P/cmdline 2>/dev/null)
case "$C" in *sd-collect/driver/collection_driver.py*depth_recall*) kill $P; echo "stopped driver $P at $(date -u)";; *) echo "pid $P is not the driver: $C"; exit 1;; esac
sleep 5; while camp; do echo "a campaign started before the stop; waiting"; sleep 30; done
rm -f $A/pids/depth_recall.pid
bash $HOME/sd-collect/launch_dr.sh
