#!/usr/bin/env bash
# New prospective SameDrawer seed bank: 500 per reset drawer from 9100000, validation seeds excluded.
source $HOME/sd-collect/env.sh
cd $HOME/sd-collect/repo
PYTHONPATH=. $SIM -m utils.collection.drawer_balance plan --output $HOME/sd-collect/driver/bank.json \
  --start-seed 9100000 --per-drawer 500 --exclude utils/collection/validation_seeds/same_drawer_v1.json
echo "exit=$?" ; date
