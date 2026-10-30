#!/usr/bin/env bash
# CabinetSearch5 profile 7 done (target 1020 accepted) -> finalize: validation100 from 7847000-7847179, exact 1000,
# README, checklist, pi0.5 readiness. No deletions, no upload (the upload is a separate step).
A=$HOME/sd-collect/driver
L0=$(grep -n "c5v7-train-001" $A/driver-cabinet_search5.out | head -1 | cut -d: -f1)
until tail -n +$L0 $A/driver-cabinet_search5.out | grep -q "target reached\|seed range exhausted"; do sleep 60; done
while P=$(cat $A/pids/cabinet_search5.pid 2>/dev/null) && [ -n "$P" ] && tr "\0" " " < /proc/$P/cmdline 2>/dev/null | grep -q collection_driver.py; do sleep 60; done
while ps -eo args | grep "[p]ostbatch.py cabinet_search5" | grep -q sd-collect; do sleep 60; done
echo "finalize start $(date -u +%FT%TZ)"
bash $HOME/sd-collect/finalize_cabinet5_ag.sh > $A/final/finalize_cabinet5.log 2>&1
echo "finalize exit $? $(date -u +%FT%TZ)"
