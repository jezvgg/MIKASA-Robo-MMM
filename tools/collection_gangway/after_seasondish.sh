#!/usr/bin/env bash
# SeasonDish AG done -> finalize (validation100, exact 1000, README, checklist, vla, rejected-RGB prune) ->
# CabinetSearch AG pilot: TARGET=1 runs two 20-seed batches with their gates, then the driver stops for review.
A=$HOME/sd-collect/driver
until grep -q "target reached\|seed range exhausted" $A/driver-season_dish.out 2>/dev/null; do sleep 60; done
while P=$(cat $A/pids/season_dish.pid 2>/dev/null) && [ -n "$P" ] && tr "\0" " " < /proc/$P/cmdline 2>/dev/null | grep -q collection_driver.py; do sleep 60; done
while ps -eo args | grep "[p]ostbatch.py season_dish" | grep -q sd-collect; do sleep 60; done
echo "finalize start $(date -u +%FT%TZ)"
bash $HOME/sd-collect/finalize_seasondish_ag.sh > $A/final/finalize_seasondish.log 2>&1
code=$?; echo "finalize exit $code $(date -u +%FT%TZ)"
[ $code -eq 0 ] || exit 1
TARGET=1 BUDGET=40 bash $HOME/sd-collect/launch_cab.sh
