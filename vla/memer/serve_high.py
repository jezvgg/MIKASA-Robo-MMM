"""MemER's high level as a local server for closed-loop evaluation (utils/policies/evaluate.py --backend memer).

Runs in the Qwen3-VL env; the authors' memer_eval comes from third_party/memer:
    ~/envs/qwen3vl/bin/python vla/memer/serve_high.py --model ~/memer-run/high-1500 \
        [--port 8100] [--system-role assistant] [--device cuda:0]

One client at a time: one evaluation shard per server and GPU. Per episode the client sends
`reset` with the task instruction, then at every low-level query `step` with the 10 Hz frames
since its previous call; the model runs once, at the newest frame (memer_eval/deploy.py,
MemERDeploymentPolicy). The context is the SFT export's (vla/memer/export_sft.sh): the three
cameras stacked to 192x576 (configs/fetch_stacked_rgb_camera_layout.json), every 5th frame
(2 Hz of the 10 Hz data), 8 recent frames, 8 memory keyframes; the memory merges keyframes 4
subsampled steps apart or closer (TRAINING_DEVIATIONS.md, Evaluation protocol). The authors'
eval sends the system prompt in a system turn, their training renders it as an assistant turn
(TRAINING_DEVIATIONS.md, chat roles): --system-role picks one, compare both.
Connections are local and need the key in $MEMER_HIGH_AUTHKEY (default "memer-high").
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAYOUT = REPO / "vla/memer/configs/fetch_stacked_rgb_camera_layout.json"
AUTHKEY_ENV = "MEMER_HIGH_AUTHKEY"  # utils/policies/evaluate.py reads the same variable


def decode_frame(frame):
    """utils/policies/evaluate.py encode_frame, reversed: camera key -> HxWx3 uint8."""
    return {key: np.frombuffer(data, np.uint8).reshape(shape) for key, (data, shape) in frame.items()}


def handle(policy, request, metadata):
    command = request.get("cmd")
    if command == "metadata":
        return metadata
    if command == "reset":
        policy.reset(instruction=request["instruction"])
        return {"ok": True}
    if command == "step":
        result = policy.step(*[decode_frame(frame) for frame in request["frames"]])
        return dataclasses.asdict(result)
    raise ValueError(f"unknown command {command!r}")


def serve(connection, policy, metadata):
    while True:
        try:
            request = connection.recv()
        except EOFError:
            return
        try:
            reply = handle(policy, request, metadata)
        except Exception as error:  # the client raises it; the server stays up for the next shard
            logging.exception("request %r failed", request.get("cmd"))
            reply = {"error": f"{type(error).__name__}: {error}"}
        connection.send(reply)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="the trained Qwen3-VL run (its root holds the processor)")
    parser.add_argument("--processor", help="default: --model")
    parser.add_argument("--system-role", choices=("system", "assistant"), default="system")
    parser.add_argument("--camera-layout", type=Path, default=LAYOUT)
    parser.add_argument("--frame-subsample", type=int, default=5)
    parser.add_argument("--recent-frames", type=int, default=8)
    parser.add_argument("--memory-length", type=int, default=8)
    parser.add_argument("--merge-distance", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation", help="default: flash_attention_2 on CUDA (the authors')")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(REPO / "third_party/memer"))
    from memer_eval.deploy import DeploymentConfig, MemERDeploymentPolicy
    from memer_eval.inference import QwenStructuredPredictor

    config = DeploymentConfig(frame_subsample=args.frame_subsample, recent_frames_length=args.recent_frames,
                              memory_length=args.memory_length, merge_distance=args.merge_distance,
                              camera_layout_config=str(args.camera_layout))
    predictor = QwenStructuredPredictor(args.model, processor_path=args.processor, system_role=args.system_role,
                                        device=args.device, dtype=args.dtype,
                                        attn_implementation=args.attn_implementation,
                                        max_new_tokens=args.max_new_tokens)
    policy = MemERDeploymentPolicy(predictor, config=config)
    metadata = {"model": predictor.model_path, "processor": str(Path(predictor.processor_path).resolve()),
                "system_role": args.system_role, "config": dataclasses.asdict(config)}
    authkey = os.environ.get(AUTHKEY_ENV, "memer-high").encode()
    with Listener((args.host, args.port), authkey=authkey) as listener:
        logging.info("Serving the MemER high level %s on %s:%d", predictor.model_path, args.host, args.port)
        while True:
            try:
                connection = listener.accept()
            except AuthenticationError:
                logging.warning("refused a client with another key")
                continue
            except (EOFError, OSError) as error:  # a client gone before the handshake
                logging.warning("dropped a connection before the handshake: %r", error)
                continue
            with connection:
                logging.info("client %s connected", listener.last_accepted)
                serve(connection, policy, metadata)


if __name__ == "__main__":
    main()
