# Environment for the SameDrawer second-1000 collection on this server (sourced by every runner).
W=$HOME/mikasa-pi05
export SIM=$W/sim-venv/bin/python EXP=$HOME/sd-collect/venv-export/bin/python
export MS_ASSET_DIR=$W/maniskill-assets CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
VK=$(bash $W/MIKASA-Robo-MMM/vla/pi05_first_frame/scripts/fix_vulkan.sh --python $SIM 2>/dev/null | tail -1)
[ -n "${VK#VK_ICD_FILENAMES=}" ] && export "$VK"
for v in $(env | grep -o "^MIKASA_[A-Z_]*"); do unset $v; done
