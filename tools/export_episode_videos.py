"""Cut single episodes out of a LeRobot v3.0 dataset's videos into ordinary mp4 files.

LeRobot v3.0 does not keep one video per episode: each camera's episodes are concatenated into
files videos/<camera>/chunk-XXX/file-YYY.mp4 (about 15 episodes per file in our datasets), and
meta/episodes/*.parquet says which file holds an episode and from which second to which. This
script reads that table, takes the matching stretch of the file and re-encodes it as H.264 yuv420p,
which QuickTime, browsers and messengers play (our files are H.264 yuv444p, which most of them do not).

    # from the Hub (downloads only meta/ and the video files the episodes need)
    python tools/export_episode_videos.py nurtayev-d/cabinetsearch-ag-1000ep --episodes 0 5 --out clips
    # from a local copy, all three cameras side by side in one clip
    python tools/export_episode_videos.py ~/datasets/cabinetsearch-ag-1000ep --episodes 0 --grid --out clips

Requires: pip install av pyarrow numpy huggingface_hub
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib

import av
import numpy as np
import pyarrow.parquet as pq


def open_dataset(source: str, revision: str | None):
    """Return (root, fetch): fetch(relative path) -> local path, downloading from the Hub if needed."""
    local = pathlib.Path(source).expanduser()
    if local.exists():
        return local, lambda rel: local / rel
    from huggingface_hub import hf_hub_download, snapshot_download

    root = pathlib.Path(snapshot_download(source, repo_type="dataset", revision=revision, allow_patterns=["meta/*"]))
    return root, lambda rel: pathlib.Path(
        hf_hub_download(source, rel, repo_type="dataset", revision=revision))


def read_frames(path: pathlib.Path, start: float, end: float, fps: float) -> list[np.ndarray]:
    """Decoded RGB frames whose timestamps fall in [start, end)."""
    half = 0.5 / fps
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(start - 1.0, 0) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            t = frame.time
            if t is None or t < start - half:
                continue
            if t >= end - half:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def write_mp4(path: pathlib.Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=int(round(fps)))
        stream.width, stream.height, stream.pix_fmt = width + width % 2, height + height % 2, "yuv420p"
        for image in frames:
            if image.shape[0] % 2 or image.shape[1] % 2:
                image = np.pad(image, ((0, image.shape[0] % 2), (0, image.shape[1] % 2), (0, 0)))
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def grid(views: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Cameras side by side, each scaled (nearest) to the tallest view's height."""
    height = max(v[0].shape[0] for v in views)
    n = min(len(v) for v in views)
    scaled = []
    for frames in views:
        k = height // frames[0].shape[0]
        scaled.append([np.repeat(np.repeat(f, k, 0), k, 1) if k > 1 else f for f in frames[:n]])
    return [np.concatenate([v[i] for v in scaled], axis=1) for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="Hub dataset id (user/name) or a local dataset directory")
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--cameras", nargs="*", help="camera names, e.g. left_base_camera_link; default all")
    parser.add_argument("--grid", action="store_true", help="one clip per episode with the cameras side by side")
    parser.add_argument("--out", default="clips")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    root, fetch = open_dataset(args.source, args.revision)
    info = json.loads((root / "meta/info.json").read_text())
    fps = float(info["fps"])
    table = pq.read_table(sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))).to_pandas()
    keys = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
    if args.cameras:
        keys = [k for k in keys if any(k.endswith(c) for c in args.cameras)]
    out = pathlib.Path(args.out)
    for episode in args.episodes:
        rows = table[table.episode_index == episode]
        if rows.empty:
            raise SystemExit(f"episode {episode} is not in the dataset ({len(table)} episodes)")
        row = rows.iloc[0]
        views = []
        for key in keys:
            rel = info["video_path"].format(video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
                                            file_index=int(row[f"videos/{key}/file_index"]))
            start, end = float(row[f"videos/{key}/from_timestamp"]), float(row[f"videos/{key}/to_timestamp"])
            frames = read_frames(fetch(rel), start, end, fps)
            print(f"episode {episode} {key}: {rel} {start:.1f}-{end:.1f} s, {len(frames)} frames "
                  f"(table length {int(row['length'])})")
            views.append(frames)
            if not args.grid:
                write_mp4(out / f"episode_{episode:06d}_{key.split('.')[-1]}.mp4", frames, fps)
        if args.grid:
            write_mp4(out / f"episode_{episode:06d}_grid.mp4", grid(views), fps)


if __name__ == "__main__":
    main()
