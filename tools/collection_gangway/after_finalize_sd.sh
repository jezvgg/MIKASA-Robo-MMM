#!/usr/bin/env bash
# Start SeasonDish AG once the SameDrawer AG finalize exited 0 (CPU: never both 8-job campaigns at once).
until grep -q "^finalize exit" $HOME/sd-collect/after_samedrawer.log 2>/dev/null; do sleep 60; done
grep -q "^finalize exit 0 " $HOME/sd-collect/after_samedrawer.log || { echo "SameDrawer finalize failed; SeasonDish not started"; exit 1; }
bash $HOME/sd-collect/launch_sdish.sh
