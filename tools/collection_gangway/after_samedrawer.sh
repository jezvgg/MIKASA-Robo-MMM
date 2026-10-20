#!/usr/bin/env bash
# Wait until the SameDrawer AG driver logged "target reached" and exited (its last gates written), then finalize.
A=$HOME/sd-collect/driver
until grep -q "target reached" $A/driver-same_drawer.out 2>/dev/null; do sleep 60; done
while P=$(cat $A/pids/same_drawer.pid 2>/dev/null) && [ -n "$P" ] && tr "\0" " " < /proc/$P/cmdline 2>/dev/null | grep -q collection_driver.py; do sleep 60; done
while ps -eo args | grep "[p]ostbatch.py same_drawer" | grep -q sd-collect; do sleep 60; done
echo "finalize start $(date -u +%FT%TZ)"
bash $HOME/sd-collect/finalize_samedrawer_ag.sh > $A/final/finalize.log 2>&1
echo "finalize exit $? $(date -u +%FT%TZ)"
