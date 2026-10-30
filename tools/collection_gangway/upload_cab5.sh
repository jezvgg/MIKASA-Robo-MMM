#!/usr/bin/env bash
# Owner 2026-10-26: CabinetSearch5 AG -> nurtayev-d/cabinetsearch5-ag-1000ep: the dataset at the repo root, then each
# training batch's run dir as sources/<batch>/ (as for the other AG datasets), straight from the server through Xet
# (the proxy blocks the LFS S3 host). The token is the owner's /dev/shm file; this script never reads it.
source $HOME/sd-collect/env.sh
export HF_HUB_DISABLE_XET=0 HF_TOKEN_PATH=/dev/shm/hf_token_nurtayev
PY=$HOME/sd-collect/venv-export/bin/python; U=$HOME/sd-collect/hf_folder_upload.py; R=$HOME/sd-collect/runs-ag
REPO=nurtayev-d/cabinetsearch5-ag-1000ep
echo "== dataset $(date -u +%FT%TZ)"
$PY $U --src $R/final/cabinetsearch5-ag-1000ep --repo $REPO --prefix "" --portion-gb 1.0 || { echo "FAILED dataset"; exit 1; }
for b in $(cd $R && ls -d cabinet_search5-c5v7-train-*); do
  echo "== $b $(date -u +%FT%TZ)"
  $PY $U --src $R/$b --repo $REPO --prefix sources/$b/ --portion-gb 1.0 || { echo "FAILED $b"; exit 1; }
done
echo "ALL DONE $(date -u +%FT%TZ)"
