"""Query a reloaded server with real dataset observations and session resets."""
import argparse
import json
from pathlib import Path

import numpy as np

from utils.policies.data import FetchDataset, CAMERAS
from utils.policies.evaluate import RoboMMEClient, HamletClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("robomme", "hamlet"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = FetchDataset(args.root)
    client = (RoboMMEClient if args.backend == "robomme" else HamletClient)(args.host, args.port)
    results = []
    for task in source.tasks:
        for episode in [e for e in source.episodes if e.task == task][:2]:
            if episode.length < 17:
                raise ValueError("probe requires at least 17 frames")
            rows = source.episode_rows(episode.global_index)
            def observation(frame):
                return {"observation/state": np.asarray(rows["observation.state"][frame], dtype=np.float32),
                    **{key: source.image(episode, frame, camera) for key, camera in zip(
                        ("observation/image", "observation/hand_image", "observation/side_image"), CAMERAS)},
                    "prompt": source.instructions[(task, int(rows["task_index"][frame]))]}
            client.reset(f"{args.output.stem}:{task}:{episode.source_index}")
            client.ingest(observation(0))
            shapes = [list(client.infer(observation(0)).shape)]
            for frame in range(1, 17):
                client.ingest(observation(frame))
            shapes.append(list(client.infer(observation(16)).shape))
            result = {"task": task, "episode": episode.source_index,
                      "real_query_frames": [0, 16], "action_shapes": shapes,
                      "history_observations": client.frames, "reset_confirmed": True}
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"backend": args.backend, "real_dataset_probe": results,
                                      "is_simulator_rollout": False}, indent=2))


if __name__ == "__main__":
    main()
