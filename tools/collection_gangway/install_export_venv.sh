#!/usr/bin/env bash
# Isolated LeRobot v3 export environment (utils/collection/requirements-lerobot.txt), CPU torch, in our folder.
UV=${UV:?set UV to a uv binary}
export UV_CACHE_DIR=$HOME/sd-collect/.uv-cache
cd $HOME/sd-collect
$UV venv --python /home/user/conda/bin/python3.11 venv-export
$UV pip install --python venv-export/bin/python --torch-backend cpu -r repo/utils/collection/requirements-lerobot.txt av==15.1.0 datasets==4.1.1
code=$?
venv-export/bin/python - <<PY
import importlib.metadata as m
for k in ("lerobot","torch","torchvision","torchcodec","av","datasets","numpy","h5py","sentencepiece"):
    try: print(k, m.version(k))
    except Exception as e: print(k, "MISSING", e)
PY
rm -rf $UV_CACHE_DIR
du -sh venv-export
echo "install exit=$code"
