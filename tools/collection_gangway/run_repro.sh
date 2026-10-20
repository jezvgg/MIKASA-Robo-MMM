#!/usr/bin/env bash
source $HOME/sd-collect/env.sh
cd $HOME/sd-collect/repo
PYTHONPATH=. $SIM -m utils.collection.campaign --profile same_drawer \
  --tokenizer $HOME/mikasa-pi05/openpi-cache/big_vision/paligemma_tokenizer.model \
  --output $HOME/sd-collect/repro --seed-manifest $HOME/sd-collect/repro_manifest.json \
  --purpose development --jobs 4 --through validated --timeout 1800
echo "exit=$?"; date
