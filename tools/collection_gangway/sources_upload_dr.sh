#!/usr/bin/env bash
# Owner 2026-10-21 «Дозагрузи оставшиеся h5»: DR profile 6 batches 035-055 -> nurtayev-d/depthrecall-ag-1000ep/sources/<batch>/,
# straight from the server through Xet (the proxy blocks the LFS S3 host). Batches 001-034 went through the Mac earlier.
source $HOME/sd-collect/env.sh
export HF_HUB_DISABLE_XET=0
PY=$HOME/sd-collect/venv-export/bin/python
for n in $(seq -w 35 55); do
  b=depth_recall-ag6-train-0$n
  echo "== $b $(date -u +%FT%TZ)"
  $PY $HOME/sd-collect/hf_folder_upload.py --src $HOME/sd-collect/runs-ag/$b --repo nurtayev-d/depthrecall-ag-1000ep --prefix sources/$b/ --portion-gb 1.0 || { echo "FAILED $b"; exit 1; }
done
echo "ALL DONE $(date -u +%FT%TZ)"
