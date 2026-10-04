#!/usr/bin/env bash
# Install everything for the SameDrawer pi0.5 first-frame baseline on a fresh GPU server.
# No root, no apt: all files go under $PI05_WORK (default ~/mikasa-pi05). Idempotent; re-run
# after a failure. Steps can be run alone: setup_server.sh [step ...]
#
# First time:
#   git clone -b feat/pi05-first-frame-samedrawer https://github.com/pa40l/MIKASA-Robo-MMM.git \
#       ~/mikasa-pi05/MIKASA-Robo-MMM
#   ~/mikasa-pi05/MIKASA-Robo-MMM/vla/pi05_first_frame/scripts/setup_server.sh preflight
#   nohup ~/mikasa-pi05/MIKASA-Robo-MMM/vla/pi05_first_frame/scripts/setup_server.sh \
#       > ~/mikasa-pi05/setup.log 2>&1 < /dev/null &
#
#   preflight  GPU, driver, disk, and every download host this setup needs
#   repo       update this MIKASA-Robo-MMM checkout (git pull --ff-only)
#   openpi     clone openpi at OPENPI_COMMIT, apply openpi.patch, install the locked
#              packages, install mikasa_pi05                     -> $OPENPI_DIR (.venv inside)
#   sim        simulator venv: requirements-sim.txt + torch 2.14.0
#              for this driver + openpi-client                   -> $SIM_VENV
#   assets     RoboCasa scenes (haosulab/RoboCasa)               -> $MS_ASSET_DIR
#   data       nurtayev-d/samedrawer-1000ep at a pinned revision -> $SAMEDRAWER_DATASET_DIR
#   weights    pi05_base params + PaliGemma tokenizer            -> $OPENPI_DATA_HOME
#              from Hugging Face mirrors (pinned revisions), else from gs://; every file must
#              match weights.sha256, the checksums of the original gs:// files
#   vulkan     find a Vulkan driver manifest that renders on the GPU
#   env        write $PI05_WORK/server.env (source it before every command)
#   clean      empty the uv cache (venvs do not need it) and report disk use
#
# Behind a package mirror (PIP_INDEX_URL/UV_INDEX_URL/UV_DEFAULT_INDEX not pypi.org) openpi's
# lockfile is exported to pinned requirements and installed through the mirror, because the
# lockfile itself points at files.pythonhosted.org.
# Every path can be overridden by exporting the variable first (useful when parts already exist).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # vla/pi05_first_frame of this checkout
PI05_WORK="${PI05_WORK:-$HOME/mikasa-pi05}"
REPO_DIR="${REPO_DIR:-$(cd "$HERE/../.." && pwd)}"
OPENPI_URL="${OPENPI_URL:-https://github.com/Physical-Intelligence/openpi.git}"
OPENPI_DIR="${OPENPI_DIR:-$PI05_WORK/openpi}"
SIM_VENV="${SIM_VENV:-$PI05_WORK/sim-venv}"
MS_ASSET_DIR="${MS_ASSET_DIR:-$PI05_WORK/maniskill-assets}"
SAMEDRAWER_DATASET_DIR="${SAMEDRAWER_DATASET_DIR:-$PI05_WORK/data/samedrawer-1000ep}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PI05_WORK/openpi-cache}"
VULKAN_DIR="${VULKAN_DIR:-$PI05_WORK/vulkan}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PI05_WORK/uv-cache}"
export HF_XET_CHUNK_CACHE_SIZE_BYTES="${HF_XET_CHUNK_CACHE_SIZE_BYTES:-0}"   # no extra copy of the dataset

HF_DATASET=nurtayev-d/samedrawer-1000ep
HF_REVISION=d126ebae7e8aa4217c2a61a5b08214fc75f1bdac
# sha256 of meta/info.json of the dataset that the configs and norm stats were made for.
INFO_SHA256=975f8983ab7f7d75a3a812c4bf3701d56e3eedf4b133ba17c6d20ce0cf3d5699
ROBOCASA_URL=https://huggingface.co/datasets/haosulab/RoboCasa/resolve/main/robocasa_dataset.zip
# Byte-identical Hugging Face copies of gs://openpi-assets/checkpoints/pi05_base/params and
# gs://big_vision/paligemma_tokenizer.model (verified against weights.sha256 on 2026-10-04).
WEIGHTS_REPO=wz7in/pi05_base
WEIGHTS_REVISION=1fff0c3949c177e314bd19622053d45be32e61a8
TOKENIZER_REPO=leo009/paligemma_tokenizer.model
TOKENIZER_REVISION=2506bd189b4713d798c92b6070f4177b590b4d44
TORCH_VERSION=2.14.0
TORCH_CU126_INDEX=https://download.pytorch.org/whl/cu126

log() { printf '\n== %s\n' "$*"; }

need_uv() {
  command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    log "installing uv into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

package_index() {  # the default package index in use
  local index="${UV_DEFAULT_INDEX:-${UV_INDEX_URL:-${PIP_INDEX_URL:-https://pypi.org/simple}}}"
  echo "${index%/}"
}

driver_major() {
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1
}

step_preflight() {
  log "preflight"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "nvidia-smi: no GPU visible"
  python3 -V || true
  need_uv; uv --version
  echo "package index: $(package_index)"
  echo "disk: $(du -sh "$HOME" 2>/dev/null | cut -f1) used in $HOME; this setup adds ~55 GB, each training checkpoint ~7 GB"
  local blocked=()
  ok()      { if "$@" >/dev/null 2>&1; then return 0; else return 1; fi; }
  check()   { local name="$1"; shift; if ok "$@"; then echo "  ok       $name"; else echo "  BLOCKED  $name"; blocked+=("$name"); fi; }
  # Any HTTP answer counts for hosts we only talk to; a proxy refusal makes curl itself fail.
  reach()   { curl -sS -o /dev/null --max-time 30 "$1" 2>/dev/null; }
  fetch()   { curl -fsSL -r 0-1023 -o /dev/null --max-time 60 "$1" 2>/dev/null; }
  check "package index $(package_index)" curl -fsS -o /dev/null --max-time 30 "$(package_index)/numpy/"
  check "github.com (code)" git ls-remote https://github.com/Physical-Intelligence/openpi.git HEAD
  check "huggingface.co + its CDN (dataset)" fetch "https://huggingface.co/datasets/$HF_DATASET/resolve/$HF_REVISION/meta/tasks.parquet"
  check "huggingface.co + its CDN (RoboCasa scenes)" fetch "$ROBOCASA_URL"
  check "huggingface.co (pi05_base weights, $WEIGHTS_REPO)" fetch "https://huggingface.co/$WEIGHTS_REPO/resolve/$WEIGHTS_REVISION/params/ocdbt.process_0/d/828bee85475e37c61e1cc19e32d1c5ef"
  check "huggingface.co (PaliGemma tokenizer, $TOKENIZER_REPO)" fetch "https://huggingface.co/$TOKENIZER_REPO/resolve/$TOKENIZER_REVISION/paligemma_tokenizer.model"
  if fetch "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"; then
    echo "  ok       storage.googleapis.com (optional: original source of the weights)"
  else
    echo "  blocked  storage.googleapis.com (optional: the weights come from Hugging Face)"
  fi
  local major; major="$(driver_major)"
  if [[ -n "$major" && "$major" -lt 580 ]]; then
    check "download.pytorch.org (torch for CUDA 12.6; driver $major < 580)" curl -fsS -o /dev/null --max-time 30 "$TORCH_CU126_INDEX/torch/"
  fi
  if reach https://api.wandb.ai/; then echo "  ok       api.wandb.ai (optional: training charts)"; else echo "  blocked  api.wandb.ai (optional: training charts)"; fi
  if (( ${#blocked[@]} )); then
    echo
    echo "Ask the server owner to allow these downloads, then run preflight again:"
    printf '  - %s\n' "${blocked[@]}"
    return 1
  fi
  echo "all downloads reachable"
}

step_repo() {
  log "repo $REPO_DIR"
  if [[ -z "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
    git -C "$REPO_DIR" pull -q --ff-only || echo "(pull failed; kept the checkout as is)"
  else
    echo "(local changes; not pulling)"
  fi
  git -C "$REPO_DIR" log --oneline -1
}

step_openpi() {
  need_uv
  local commit
  commit="$(cat "$HERE/OPENPI_COMMIT")"
  log "openpi $commit + openpi.patch -> $OPENPI_DIR"
  if [[ ! -d "$OPENPI_DIR/.git" ]]; then
    git clone -q "$OPENPI_URL" "$OPENPI_DIR"
  fi
  if [[ "$(git -C "$OPENPI_DIR" rev-parse HEAD)" != "$commit" ]]; then
    git -C "$OPENPI_DIR" fetch -q origin "$commit" 2>/dev/null || true
    git -C "$OPENPI_DIR" checkout -q "$commit"
  fi
  if git -C "$OPENPI_DIR" apply --reverse --check "$HERE/openpi.patch" 2>/dev/null; then
    echo "patch already applied"
  else
    git -C "$OPENPI_DIR" apply "$HERE/openpi.patch"
  fi
  local mode="${OPENPI_INSTALL:-}"
  if [[ -z "$mode" ]]; then
    [[ "$(package_index)" == *pypi.org* ]] && mode=sync || mode=export
  fi
  if [[ "$mode" == sync ]]; then
    (cd "$OPENPI_DIR" && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen)
  else
    echo "installing the locked versions through $(package_index)"
    (cd "$OPENPI_DIR" \
      && uv export --frozen --no-hashes --format requirements-txt -o requirements-locked.txt >/dev/null \
      && { [[ -x .venv/bin/python ]] || uv venv --python 3.11 .venv; } \
      && GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python -r requirements-locked.txt)
  fi
  uv pip install --python "$OPENPI_DIR/.venv/bin/python" --no-deps -e "$HERE"
  "$OPENPI_DIR/.venv/bin/python" -c "import jax, openpi, mikasa_pi05; print('jax', jax.__version__, 'devices', jax.devices())"
}

step_sim() {
  need_uv
  log "simulator venv -> $SIM_VENV"
  [[ -x "$SIM_VENV/bin/python" ]] || uv venv --python 3.11 "$SIM_VENV"
  # The PyPI torch 2.14.0 wheel needs CUDA 13 (driver >= 580). Older drivers get the same
  # release built for CUDA 12.6; eval_samedrawer.py accepts only that local build tag.
  # torch goes in the same resolve, so mani-skill's unpinned torch requirement cannot pull another.
  local major torch_args
  major="$(driver_major)"
  if [[ -n "$major" && "$major" -lt 580 ]]; then
    echo "driver $major < 580: torch $TORCH_VERSION+cu126"
    torch_args=("torch==$TORCH_VERSION+cu126" --index "$TORCH_CU126_INDEX" --index-strategy unsafe-best-match)
  else
    torch_args=("torch==$TORCH_VERSION")
  fi
  uv pip install --python "$SIM_VENV/bin/python" -r "$HERE/requirements-sim.txt" \
    --override "$HERE/overrides-sim.txt" "${torch_args[@]}"
  if [[ -d "$OPENPI_DIR/packages/openpi-client" ]]; then
    uv pip install --python "$SIM_VENV/bin/python" --no-deps "$OPENPI_DIR/packages/openpi-client"
  else
    echo "run the openpi step first (openpi-client comes from the openpi checkout)" >&2
    return 1
  fi
  "$SIM_VENV/bin/python" -c "import torch, mani_skill, sapien; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mani_skill', mani_skill.__version__)"
}

step_assets() {
  log "RoboCasa assets -> $MS_ASSET_DIR"
  local target="$MS_ASSET_DIR/data/scene_datasets"
  if [[ -f "$MS_ASSET_DIR/.robocasa_complete" ]]; then
    echo "already present"
    return
  fi
  mkdir -p "$target"
  local zip="$MS_ASSET_DIR/robocasa_dataset.zip"
  if [[ ! -f "$zip" ]]; then
    curl -L --fail -C - -o "$zip.partial" "$ROBOCASA_URL"
    mv "$zip.partial" "$zip"
  fi
  "$SIM_VENV/bin/python" -m zipfile -e "$zip" "$target"   # unzip may be missing without apt
  rm -f "$zip"
  touch "$MS_ASSET_DIR/.robocasa_complete"
}

step_data() {
  log "dataset $HF_DATASET@${HF_REVISION:0:7} -> $SAMEDRAWER_DATASET_DIR"
  local args=(download "$HF_DATASET" --repo-type dataset --revision "$HF_REVISION" --local-dir "$SAMEDRAWER_DATASET_DIR")
  if ! "$SIM_VENV/bin/hf" "${args[@]}" >/dev/null; then
    echo "xet transfer failed; retrying over plain HTTPS"
    HF_HUB_DISABLE_XET=1 "$SIM_VENV/bin/hf" "${args[@]}" >/dev/null
  fi
  local sha
  sha="$(sha256sum "$SAMEDRAWER_DATASET_DIR/meta/info.json" | cut -d' ' -f1)"
  if [[ "$sha" != "$INFO_SHA256" ]]; then
    echo "meta/info.json sha256 $sha differs from the expected $INFO_SHA256" >&2
    return 1
  fi
  echo "ok: $(du -sh "$SAMEDRAWER_DATASET_DIR" | cut -f1)"
}

weights_ok() {
  (cd "$OPENPI_DATA_HOME" && grep -v '^#' "$HERE/weights.sha256" | sha256sum -c --quiet) 2>/dev/null
}

step_weights() {
  # openpi looks for gs://X/Y at $OPENPI_DATA_HOME/X/Y, so files placed there are used as is.
  log "pi05_base weights and tokenizer -> $OPENPI_DATA_HOME"
  if weights_ok; then echo "already present and verified"; return; fi
  local hf="$SIM_VENV/bin/hf"
  mkdir -p "$OPENPI_DATA_HOME"
  if ! { "$hf" download "$WEIGHTS_REPO" --revision "$WEIGHTS_REVISION" --include "params/*" \
           --local-dir "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base" >/dev/null \
         || HF_HUB_DISABLE_XET=1 "$hf" download "$WEIGHTS_REPO" --revision "$WEIGHTS_REVISION" --include "params/*" \
           --local-dir "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base" >/dev/null; } \
     || ! "$hf" download "$TOKENIZER_REPO" paligemma_tokenizer.model --revision "$TOKENIZER_REVISION" \
           --local-dir "$OPENPI_DATA_HOME/big_vision" >/dev/null; then
    echo "Hugging Face mirrors failed; trying the original gs:// bucket"
    OPENPI_DATA_HOME="$OPENPI_DATA_HOME" "$OPENPI_DIR/.venv/bin/python" -c '
from openpi.shared import download
download.maybe_download("gs://openpi-assets/checkpoints/pi05_base/params", token="anon")
download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
'
  fi
  if ! weights_ok; then
    (cd "$OPENPI_DATA_HOME" && grep -v '^#' "$HERE/weights.sha256" | sha256sum -c) | grep -v ': OK$' >&2 || true
    echo "the weights do not match the original pi05_base checksums" >&2
    return 1
  fi
  echo "verified: $(du -sh "$OPENPI_DATA_HOME" | cut -f1)"
}

step_vulkan() {
  log "Vulkan"
  local line
  line="$("$HERE/scripts/fix_vulkan.sh" --python "$SIM_VENV/bin/python" --icd-dir "$VULKAN_DIR" | tee /dev/stderr | tail -1)"
  VK_LINE="$line"
}

step_env() {
  log "writing $PI05_WORK/server.env"
  local vk="${VK_LINE:-}"
  if [[ -z "$vk" && -f "$VULKAN_DIR/nvidia_egl_icd.json" ]]; then
    vk="VK_ICD_FILENAMES=$VULKAN_DIR/nvidia_egl_icd.json"
  fi
  mkdir -p "$PI05_WORK"
  cat > "$PI05_WORK/server.env" <<EOF
# Source before every command: . $PI05_WORK/server.env
export PI05_WORK="$PI05_WORK"
export REPO_DIR="$REPO_DIR"
export OPENPI_DIR="$OPENPI_DIR"
export SIM_VENV="$SIM_VENV"
export MS_ASSET_DIR="$MS_ASSET_DIR"
export SAMEDRAWER_DATASET_DIR="$SAMEDRAWER_DATASET_DIR"
export OPENPI_DATA_HOME="$OPENPI_DATA_HOME"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export PATH="\$HOME/.local/bin:\$PATH"
EOF
  if [[ "$vk" == VK_ICD_FILENAMES=?* ]]; then
    echo "export $vk" >> "$PI05_WORK/server.env"
  fi
  cat "$PI05_WORK/server.env"
}

step_clean() {
  log "clean"
  need_uv
  uv cache clean >/dev/null 2>&1 || true
  echo "disk: $(du -sh "$HOME" 2>/dev/null | cut -f1) used in $HOME"
}

STEPS=("$@")
[[ ${#STEPS[@]} -gt 0 ]] || STEPS=(preflight repo openpi sim assets data weights vulkan env clean)
mkdir -p "$PI05_WORK"
for s in "${STEPS[@]}"; do
  "step_$s"
done
[[ ${#STEPS[@]} -gt 1 ]] && log "setup finished: . $PI05_WORK/server.env" || true
