#!/usr/bin/env bash
# Find a Vulkan driver manifest (ICD) on which SAPIEN renders on the NVIDIA GPU.
#
# Some containers bind-mount a read-only /etc/vulkan/icd.d/nvidia_icd.json
# pointing at libGLX_nvidia.so.0, which fails vkCreateInstance without an X
# server. The same driver works headless through libEGL_nvidia.so.0. When
# /usr/share/vulkan/icd.d/nvidia_icd.json is absent, SAPIEN also substitutes
# its own GLX manifest (sapien/_vulkan_tricks.py), so the default fails twice.
#
# Usage: fix_vulkan.sh [--python PY] [--icd-dir DIR] [--install-system]
#   --python          interpreter with sapien installed (default: $PYTHON or python3)
#   --icd-dir         where to write the EGL manifest (default: $HOME/.local/share/mikasa-vulkan)
#   --install-system  also write the EGL manifest to /usr/share/vulkan/icd.d/nvidia_icd.json
#                     when that file does not exist (root only), so no variable is needed
#
# The last line of output is `VK_ICD_FILENAMES=<path>` (empty when the default works).
# Exit status is non-zero when no manifest renders on the GPU.
set -euo pipefail

PY="${PYTHON:-python3}"
ICD_DIR="${HOME}/.local/share/mikasa-vulkan"
INSTALL_SYSTEM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PY="$2"; shift 2 ;;
    --icd-dir) ICD_DIR="$2"; shift 2 ;;
    --install-system) INSTALL_SYSTEM=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

SYSTEM_ICD=/etc/vulkan/icd.d/nvidia_icd.json
SHARED_ICD=/usr/share/vulkan/icd.d/nvidia_icd.json

probe() {  # probe <manifest or empty>
  local icd="$1"
  env ${icd:+VK_ICD_FILENAMES="$icd"} timeout 180 "$PY" -W ignore -c '
import numpy as np, sapien
device = sapien.Device("cuda")
scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(device)])
scene.add_ground(0)
scene.set_ambient_light([0.5, 0.5, 0.5])
camera = scene.add_camera("probe", 64, 64, 1.0, 0.01, 10)
camera.set_pose(sapien.Pose([-1, 0, 0.5]))
scene.update_render()
camera.take_picture()
rgb = np.asarray(camera.get_picture("Color"))
assert rgb.shape == (64, 64, 4) and np.isfinite(rgb).all()
print("render ok on", device)
' >/dev/null 2>&1
}

api_version() {
  local version=""
  for path in "$SYSTEM_ICD" "$SHARED_ICD"; do
    if [[ -f "$path" ]]; then
      version=$(sed -n 's/.*"api_version" *: *"\([0-9.]*\)".*/\1/p' "$path" | head -1)
      [[ -n "$version" ]] && break
    fi
  done
  echo "${version:-1.3.0}"
}

write_egl_manifest() {  # write_egl_manifest <path>
  mkdir -p "$(dirname "$1")"
  printf '{"file_format_version": "1.0.1", "ICD": {"library_path": "libEGL_nvidia.so.0", "api_version": "%s"}}\n' \
    "$(api_version)" > "$1"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found: no NVIDIA driver visible" >&2
  exit 1
fi
echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)," \
  "capabilities: ${NVIDIA_DRIVER_CAPABILITIES:-unset}"

if [[ -n "${VK_ICD_FILENAMES:-}" ]] && probe "$VK_ICD_FILENAMES"; then
  echo "current VK_ICD_FILENAMES works"
  echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
  exit 0
fi
if (unset VK_ICD_FILENAMES; probe ""); then
  echo "default Vulkan driver works"
  echo "VK_ICD_FILENAMES="
  exit 0
fi

EGL_ICD="$ICD_DIR/nvidia_egl_icd.json"
write_egl_manifest "$EGL_ICD"
if ! probe "$EGL_ICD"; then
  cat >&2 <<'EOF'
No Vulkan driver renders on the NVIDIA GPU (neither the default nor libEGL_nvidia).
Check that the container exposes the `graphics` driver capability
(NVIDIA_DRIVER_CAPABILITIES should include graphics or be `all`).
Fallback: Mesa lavapipe (apt install mesa-vulkan-drivers) renders on the CPU;
it is slow and its pixels differ from the NVIDIA renders the dataset was made with.
EOF
  exit 1
fi
echo "default driver fails, libEGL_nvidia works: $EGL_ICD"

if [[ "$INSTALL_SYSTEM" == 1 ]]; then
  if [[ -e "$SHARED_ICD" ]]; then
    echo "$SHARED_ICD already exists; not replacing it" >&2
  elif mkdir -p "$(dirname "$SHARED_ICD")" 2>/dev/null && cp "$EGL_ICD" "$SHARED_ICD" 2>/dev/null; then
    if (unset VK_ICD_FILENAMES; probe ""); then
      echo "installed $SHARED_ICD"
      echo "VK_ICD_FILENAMES="
      exit 0
    fi
    rm -f "$SHARED_ICD"
    echo "system manifest did not help; use the explicit path" >&2
  else
    echo "cannot write $SHARED_ICD; use the explicit path" >&2
  fi
fi
echo "VK_ICD_FILENAMES=$EGL_ICD"
