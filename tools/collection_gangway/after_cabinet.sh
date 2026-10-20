#!/usr/bin/env bash
# CabinetSearch AG done -> finalize (validation100 from 7845000-7845179, exact 1000 D6-passing, README, checklist, vla,
# rejected-RGB prune). DR resumes by hand after the Cabinet upload and its local deletion (owner: DR last).
A=$HOME/sd-collect/driver
until grep -q "target reached\|seed range exhausted" <(tail -n +$(( $(grep -n "ag-train-004" $A/driver-cabinet_search.out | head -1 | cut -d: -f1) )) $A/driver-cabinet_search.out) 2>/dev/null; do sleep 60; done
while P=$(cat $A/pids/cabinet_search.pid 2>/dev/null) && [ -n "$P" ] && tr "\0" " " < /proc/$P/cmdline 2>/dev/null | grep -q collection_driver.py; do sleep 60; done
while ps -eo args | grep "[p]ostbatch.py cabinet_search" | grep -q sd-collect; do sleep 60; done
echo "finalize start $(date -u +%FT%TZ)"
bash $HOME/sd-collect/finalize_cabinet_ag.sh > $A/final/finalize_cabinet.log 2>&1
echo "finalize exit $? $(date -u +%FT%TZ)"
