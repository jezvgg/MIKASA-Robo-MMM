#!/usr/bin/env bash
# DepthRecall profile 6 done (target 1000) -> finalize: validation100 from 7965000-7965179, exact 1000, README,
# checklist, pi0.5 readiness. No deletions (owner 2026-10-19).
A=$HOME/sd-collect/driver
L0=$(grep -n "ag6-train-001" $A/driver-depth_recall.out | head -1 | cut -d: -f1)
until tail -n +$L0 $A/driver-depth_recall.out | grep -q "target reached\|seed range exhausted"; do sleep 60; done
while P=$(cat $A/pids/depth_recall.pid 2>/dev/null) && [ -n "$P" ] && tr "\0" " " < /proc/$P/cmdline 2>/dev/null | grep -q collection_driver.py; do sleep 60; done
while ps -eo args | grep "[p]ostbatch.py depth_recall" | grep -q sd-collect; do sleep 60; done
echo "finalize start $(date -u +%FT%TZ)"
bash $HOME/sd-collect/finalize_dr6_ag.sh > $A/final/finalize_dr6.log 2>&1
echo "finalize exit $? $(date -u +%FT%TZ)"
