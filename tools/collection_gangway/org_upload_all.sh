#!/usr/bin/env bash
# Upload the four AG datasets into folders of MIKASA-Robo-MMM/MIKASA-Robo-MMM, one after another, from the verified
# current builds (SeasonDish is the shaking one, CabinetSearch the five-compartment one); the folder names
# ~/datasets copies. The owner starts this with HF_TOKEN in the environment (read -rs, never written to disk); this
# script and hf_folder_upload.py never read, print or store it. Each run resumes from ~/sd-collect/org-upload/.
source $HOME/sd-collect/env.sh
PY=$HOME/sd-collect/venv-export/bin/python
export HF_HUB_DISABLE_XET=0  # 10-21: the proxy blocks the LFS S3 host; Xet (cas-server.xethub.hf.co) passes
for pair in samedrawer:samedrawer-ag-1000ep seasondish:seasondish-shake-ag-1000ep cabinetsearch:cabinetsearch5-ag-1000ep depthrecall:depthrecall-ag-1000ep; do
  folder=${pair%%:*}; name=${pair#*:}
  echo "== $folder from ~/datasets/$name $(date -u +%FT%TZ)"
  $PY $HOME/sd-collect/hf_folder_upload.py --src $HOME/datasets/$name --repo MIKASA-Robo-MMM/MIKASA-Robo-MMM --prefix $folder/ --portion-gb 1.0 || { echo "FAILED $folder $(date -u +%FT%TZ)"; exit 1; }
done
echo "ALL DONE $(date -u +%FT%TZ)"
