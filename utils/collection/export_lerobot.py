"""Export successful CabinetSearch RGB recordings with the genuine LeRobot v3 API."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

from .contract import (ACTION_NAMES, ACTION_REPEAT, CAMERAS, POLICY_FPS, ROBOT,
                       episode_summary, read_json, recording_info, write_json)


def sources(root, seeds=None):
    report = read_json(root / "attempts.json")
    selected = report["ready_seeds"] if seeds is None else seeds
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("A nonempty unique set of ready seeds is required")
    if not set(selected) <= set(report["ready_seeds"]):
        raise ValueError("Cannot export failed or unfinished RGB episodes")
    for seed in selected:
        path = root / "rgb" / str(seed) / "trajectory.h5"
        meta = recording_info(path, stage="rgb")
        if meta["episodes"][0]["episode_seed"] != seed:
            raise ValueError("Source seed disagrees with episode metadata")
        yield seed, path, meta


def frame_records(path, instruction):
    """The observation immediately before the paired action, every 0.1 seconds."""
    with h5py.File(path, "r") as h5:
        traj = h5["traj_0"]
        n = len(traj["actions"])
        for t in range(0, n, ACTION_REPEAT):
            qpos = traj["obs/agent/qpos"][t].astype(np.float32)
            yield {
                "observation.state": qpos[3:],
                "global_state": qpos[:3],
                "action": traj["actions"][t].astype(np.float32),
                "next.reward": np.array([traj["rewards"][t:t+ACTION_REPEAT].sum()], dtype=np.float32),
                "next.success": np.array([traj["success"][t+ACTION_REPEAT-1]], dtype=bool),
                **{f"observation.images.{name}": traj[f"obs/sensor_data/{name}/rgb"][t]
                   for name in CAMERAS},
                "task": instruction,
            }


def dataset_class():
    if importlib.metadata.version("lerobot") != "0.4.3":
        raise RuntimeError("Use the isolated export environment with lerobot==0.4.3")
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.datasets.video_utils import encode_video_frames
    if CODEBASE_VERSION != "v3.0":
        raise RuntimeError("The installed LeRobot does not write v3.0")

    class Dataset(LeRobotDataset):
        # The pinned upstream API exposes codec but not quality in create().
        # Keep the original writer/metadata, set high-quality RGB video encoding.
        def _encode_temporary_episode_video(self, video_key, episode_index):
            images = self._get_image_file_dir(episode_index, video_key)
            target = Path(tempfile.mkdtemp(dir=self.root)) / f"{video_key}_{episode_index}.mp4"
            encode_video_frames(images, target, self.fps, vcodec="h264", crf=18,
                                pix_fmt="yuv420p", g=2, overwrite=False)
            shutil.rmtree(images)
            return target
    return Dataset


def export(root, output, repo_id, tokenizer, seeds=None):
    import sentencepiece as spm
    entries = list(sources(root, seeds))
    run = read_json(root / "run.json")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    names = entries[0][2]["mikasa_data"]["state_joint_names"]
    if len(names) != 15:
        raise ValueError("Missing original joint order")
    features = {
        "observation.state": dict(dtype="float32", shape=(12,), names=names[3:]),
        "global_state": dict(dtype="float32", shape=(3,), names=names[:3]),
        "action": dict(dtype="float32", shape=(13,), names=ACTION_NAMES),
        "next.reward": dict(dtype="float32", shape=(1,), names=["sum_over_control_interval"]),
        "next.success": dict(dtype="bool", shape=(1,), names=["success_after_interval"]),
        **{f"observation.images.{name}": dict(dtype="video", shape=shape,
             names=["height", "width", "channels"]) for name, shape in CAMERAS.items()},
    }
    Dataset = dataset_class()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset {output}")
    dataset = Dataset.create(repo_id=repo_id, root=output, fps=POLICY_FPS,
                             robot_type=ROBOT, features=features, use_videos=True,
                             image_writer_threads=4, video_backend="pyav", vcodec="h264")
    mapping = []
    try:
        for i, (seed, path, meta) in enumerate(entries):
            instruction = meta["mikasa_data"]["instruction"]
            tokens = len(sp.encode(instruction.strip() + "\n", add_bos=True))
            if tokens > 100 or meta["mikasa_data"]["state_joint_names"] != names:
                raise ValueError("Instruction or joint order violates contract")
            for frame in frame_records(path, instruction):
                dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=False)
            with h5py.File(path, "r") as h5:
                trajectory = h5["traj_0"]
                n = len(trajectory["actions"])
                summary = episode_summary(trajectory)
                mapping.append(dict(episode_index=i, scene_seed=seed,
                    planner_seed=seed, waypoint_noise_seed=seed + run["signature"]["profile"]["planner"]["noise_seed_offset"],
                    source_h5=str(path), control_steps=n, frames=n // 2,
                    duration_seconds=summary["duration_seconds"], success=summary["success"],
                    success_once=summary["success_once"],
                    reward_sum=summary["reward_sum"], instruction=instruction,
                    instruction_tokens=tokens,
                    source_sha256=run["signature"]["code_sha256"]))
            print(f"exported seed={seed} frames={n//2}", flush=True)
    finally:
        dataset.finalize()
        dataset.stop_image_writer()
    metadata = dict(version=1, format="LeRobotDataset-v3.0", repository_id=repo_id,
        source_run=run, source_attempts=read_json(root / "attempts.json"), episodes=mapping,
        control_hz=20, policy_hz=10, action_repeat=2,
        resampling="first_target_two_step_hold_physically_validated",
        policy_inputs=["observation.state", "task"] + [f"observation.images.{name}" for name in CAMERAS],
        debug_only_fields=["global_state"], reward_semantics="sum of two 20Hz sparse rewards per 10Hz interval",
        video=dict(codec="h264", crf=18, pixel_format="yuv420p", decoded_format="RGB", gop=2),
        tokenizer_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        packages={name: importlib.metadata.version(name) for name in ("lerobot", "datasets", "torch", "av", "h5py", "sentencepiece")})
    write_json(output / "source_h5_metadata.json", metadata)
    verify(output)


def verify(output):
    """Read every numerical sample and representative RGB frames with LeRobot."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    metadata = read_json(output / "source_h5_metadata.json")
    dataset = LeRobotDataset(repo_id=metadata["repository_id"], root=output, video_backend="pyav")
    if dataset.meta.info["codebase_version"] != "v3.0" or dataset.fps != 10:
        raise ValueError("Dataset format/frequency does not match contract")
    if dataset.num_episodes != len(metadata["episodes"]):
        raise ValueError("Missing episode metadata")
    checks, offset = [], 0
    table = dataset.hf_dataset
    for episode in metadata["episodes"]:
        n = episode["frames"]
        with h5py.File(episode["source_h5"], "r") as h5:
            traj = h5["traj_0"]
            summary = episode_summary(traj)
            for key in ("control_steps", "duration_seconds", "reward_sum", "success", "success_once"):
                if episode.get(key) != summary[key]:
                    raise ValueError(f"Exported {key} metadata disagrees with H5")
            expected = {
                "action": traj["actions"][::2],
                "observation.state": traj["obs/agent/qpos"][::2][:-1, 3:],
                "global_state": traj["obs/agent/qpos"][::2][:-1, :3],
            }
            rows = table.select(range(offset, offset+n))
            for key, array in expected.items():
                np.testing.assert_array_equal(np.stack([np.asarray(v) for v in rows[key]]), array)
            timestamps = np.asarray([float(v) for v in rows["timestamp"]])
            np.testing.assert_allclose(timestamps, np.arange(n) / 10, atol=1e-4, rtol=0)
            image_checks = []
            for t in sorted({0, n//4, n//2, 3*n//4, n-1}):
                sample = dataset[offset+t]
                if sample["task"] != episode["instruction"]:
                    raise ValueError("Decoded instruction changed")
                for camera, shape in CAMERAS.items():
                    image = sample[f"observation.images.{camera}"].numpy().transpose(1, 2, 0)
                    if image.shape != shape or not np.isfinite(image).all():
                        raise ValueError(f"Invalid decoded image for {camera}")
                    source = traj[f"obs/sensor_data/{camera}/rgb"][2*t].astype(np.float32)
                    mae = float(np.abs(image * 255 - source).mean())
                    if mae > 12:
                        raise ValueError(f"Decoded camera/frame mismatch: {camera}, MAE={mae}")
                    image_checks.append(dict(frame=t, camera=camera, mean_absolute_error_255=mae))
        checks.append(dict(seed=episode["scene_seed"], frames=n, rgb_checks=image_checks))
        offset += n
    if len(dataset) != offset:
        raise ValueError("Dataset has extra or missing frames")
    write_json(output / "readback.json", dict(status="success", episodes=len(checks), frames=offset,
                                              all_numeric_samples_checked=True, results=checks))
    print(f"LeRobot v3 readback: {len(checks)} episodes, {offset} frames", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="mikasa-local/cabinet-search")
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    export(args.input.resolve(), args.output.resolve(), args.repo_id, args.tokenizer)


if __name__ == "__main__":
    main()
