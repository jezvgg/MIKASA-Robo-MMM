"""Load a HAMLET policy (GR00T N1.6 + memory head) so that it runs, not just imports.

The HAMLET repository (`third_party/hamlet`, `myungkyuKoo/HAMLET-Isaac-GR00T@63b4055`)
adds two parameter groups to GR00T N1.6: `backbone.moment_tokens` and
`action_head.memory_transformer.*`. The published base checkpoint
(`nvidia/GR00T-N1.6-3B`) does not carry them, and `from_pretrained` leaves the new
parameters at whatever `torch.empty` handed out — measured on this machine, seven of
them came back non-finite and the action head then produced NaN for every channel.
Training initialises them; inference against the base checkpoint has to do the same.

So this loader initialises exactly the parameters that are missing from the checkpoint
file, and leaves every loaded weight untouched. A fine-tuned HAMLET checkpoint
(`.../checkpoint-N`) carries the memory weights, so nothing is re-initialised there.

Usage:
    python -m utils.policies.hamlet_runtime --checkpoint $HOME/models/GR00T-N1.6-3B
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

HAMLET_ROOT = Path(
    os.environ.get(
        "HAMLET_ROOT",
        Path.home() / "policy-integration-20261019" / "third_party" / "hamlet",
    )
)
DEFAULT_CHECKPOINT = Path.home() / "models" / "GR00T-N1.6-3B"
# The RoboMME/Fetch motion is closest to this embodiment: Panda arm, joint-space
# action, base motion — the one GR00T N1.6 ships statistics for.
EMBODIMENT = "robocasa_panda_omron"
VIDEO_KEYS = ("res256_image_side_0", "res256_image_side_1", "res256_image_wrist_0")
LANGUAGE_KEY = "annotation.human.action.task_description"


def _import_gr00t():
    import sys

    if str(HAMLET_ROOT) not in sys.path:
        sys.path.insert(0, str(HAMLET_ROOT))
    import gr00t.model  # noqa: F401  (registers Gr00tN1d6 + its processor)
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    return EmbodimentTag, Gr00tPolicy


def checkpoint_parameter_names(checkpoint: Path) -> set[str]:
    """Parameter names actually saved in `checkpoint` (safetensors index or shard)."""
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        return set(json.loads(index.read_text())["weight_map"])
    from safetensors import safe_open

    names: set[str] = set()
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            names.update(handle.keys())
    return names


def initialise_missing(policy, checkpoint: Path, *, seed: int = 0) -> list[str]:
    """Give the parameters the checkpoint lacks a small deterministic value."""
    saved = checkpoint_parameter_names(checkpoint)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    touched: list[str] = []
    with torch.no_grad():
        for name, param in policy.model.named_parameters():
            if name in saved and torch.isfinite(param).all():
                continue
            if param.dim() >= 2:
                fan_in = param.shape[1]
                std = 1.0 / max(1.0, float(fan_in)) ** 0.5
                noise = torch.randn(param.shape, generator=generator, dtype=torch.float32)
                param.copy_(noise.to(param.dtype) * std)
            else:
                # Norm scales start neutral; biases start at zero.
                param.fill_(1.0 if "norm" in name else 0.0)
            touched.append(name)
    return touched


def observation(form, *, height: int = 256, width: int = 256) -> dict:
    """One dummy-but-valid RoboMME-shaped observation for a single forward pass."""
    state_dims = form["state_dims"]
    return {
        "video": {key: np.full((1, 1, height, width, 3), 128, np.uint8) for key in VIDEO_KEYS},
        "state": {key: np.zeros((1, 1, dim), np.float32) for key, dim in state_dims.items()},
        "language": {LANGUAGE_KEY: [["pick up the cup"]]},
    }


def state_dims(checkpoint: Path) -> dict[str, int]:
    stats = json.loads((checkpoint / "statistics.json").read_text())[EMBODIMENT]["state"]
    return {key: len(value["mean"]) for key, value in stats.items()}


def load_policy(checkpoint: Path | str = DEFAULT_CHECKPOINT, *, device: str = "cuda:0"):
    """Return (policy, touched_parameter_names) ready for `policy.get_action`."""
    checkpoint = Path(checkpoint)
    EmbodimentTag, Gr00tPolicy = _import_gr00t()
    policy = Gr00tPolicy(
        embodiment_tag=getattr(EmbodimentTag, EMBODIMENT.upper()),
        model_path=str(checkpoint),
        device=device,
    )
    return policy, initialise_missing(policy, checkpoint)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    policy, touched = load_policy(args.checkpoint, device=args.device)
    print(f"initialised {len(touched)} parameter(s) missing from the checkpoint", flush=True)
    action, _info = policy.get_action(
        observation({"state_dims": state_dims(args.checkpoint)})
    )
    ok = True
    for key, value in action.items():
        array = np.asarray(value)
        finite = bool(np.isfinite(array).all())
        ok = ok and finite
        print(
            f"{key}: shape={array.shape} finite={finite} "
            f"range=[{np.nanmin(array):.3f}, {np.nanmax(array):.3f}]"
        )
    print("HAMLET forward ok" if ok else "HAMLET forward produced non-finite actions")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
