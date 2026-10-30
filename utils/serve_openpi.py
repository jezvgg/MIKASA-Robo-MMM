"""Serve a trained Fetch Pi0.5 checkpoint over OpenPI's WebSocket protocol."""

from __future__ import annotations

import argparse

from openpi.policies.policy_config import create_trained_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

from utils.openpi_fetch import make_train_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id", required=True, help="same HF dataset repo_id used for training"
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint step directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument(
        "--lora", action="store_true", help="use if checkpoint was trained with --lora"
    )
    args = parser.parse_args()

    config = make_train_config(
        args.repo_id,
        "serve",
        action_horizon=args.action_horizon,
        lora=args.lora,
    )
    policy = create_trained_policy(config, args.checkpoint)
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    print(f"serving {args.checkpoint} on port {args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
