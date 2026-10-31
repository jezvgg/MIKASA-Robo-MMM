"""Run MemER's high-level policy on an observation history, from this repository.

MemER is not vendored in `.venv-sim`: it pins `lerobot==0.4.4` and
`transformers>=4.57.0` for the eval extra, which conflict with the simulation stack.
`vla/memer/setup_memer.sh` is the training-side recipe; this module is the runtime
counterpart and expects that environment:

    ~/memer-env/bin/python -m utils.policies.memer_runtime --model $HOME/models/Qwen3-VL-4B-Instruct

Two constraints are load-bearing, both measured:

* `QwenStructuredPredictor` resolves its argument with `Path(model_path).resolve()`,
  so a hub id (`Qwen/Qwen3-VL-4B-Instruct`) is not accepted — pass a local directory.
* A history is a list of per-step dicts keyed by camera name. A list of arrays (the
  shape the low-level datasets use) raises
  `ValueError: Multi-camera deployment expects each observation to be a dict keyed by camera name.`
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

MEMER_ROOT = Path(
    os.environ.get(
        "MEMER_ROOT",
        Path.home() / "policy-integration-20261019" / "third_party" / "memer",
    )
)
DEFAULT_MODEL = Path.home() / "models" / "Qwen3-VL-4B-Instruct"
# The two Fetch cameras the MemER training runs use (`vla/memer/configs`).
CAMERAS = ("left_base_camera_link", "fetch_hand")
VIEW_WIDTH, VIEW_HEIGHT = 224, 224


def load_predictor(model_path: Path | str = DEFAULT_MODEL, *, device: str = "cuda:0",
                   dtype: str = "bfloat16", max_new_tokens: int = 48):
    """Return a `QwenStructuredPredictor` for the local Qwen3-VL checkpoint."""
    if str(MEMER_ROOT) not in sys.path:
        sys.path.insert(0, str(MEMER_ROOT))
    from memer_eval.inference import QwenStructuredPredictor

    return QwenStructuredPredictor(
        str(model_path),
        device=device,
        dtype=dtype,
        attn_implementation="sdpa",
        max_new_tokens=max_new_tokens,
    )


def history(steps: int = 4, *, height: int = 240, width: int = 320, seed: int = 0) -> list[dict]:
    """A dummy observation history: `steps` frames, one grey image per camera."""
    frames = [
        {camera: np.full((height, width, 3), 128, np.uint8) for camera in CAMERAS}
        for _ in range(steps)
    ]
    rng = np.random.default_rng(seed)
    for frame in frames:
        for camera in CAMERAS:
            frame[camera] = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    return frames


def predict(frames: list[dict], *, instruction: str, predictor=None,
            cameras: tuple[str, ...] = CAMERAS):
    """Run the high-level step: subtask label plus predicted keyframe positions."""
    if predictor is None:
        predictor = load_predictor()
    from memer_eval.deploy import DeploymentConfig, predict_subtask_from_observation_history

    config = DeploymentConfig(
        camera_keys=list(cameras), view_width=VIEW_WIDTH, view_height=VIEW_HEIGHT
    )
    return predict_subtask_from_observation_history(
        frames, predictor=predictor, instruction=instruction, config=config
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="local Qwen3-VL directory (not a hub id)")
    parser.add_argument("--instruction", default="pick up the cup")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    predictor = load_predictor(args.model, device=args.device)
    print(f"model loaded: {args.model}", flush=True)
    result = predict(history(args.steps), instruction=args.instruction, predictor=predictor)
    print(f"subtask={result.current_subtask!r} parse_ok={result.parse_ok} "
          f"keyframes={result.predicted_keyframe_positions}")
    print(f"raw={result.raw_text[:160]!r}")
    ok = bool(result.parse_ok) and bool(result.current_subtask)
    print("memer forward ok" if ok else "memer returned an unparsable answer")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
