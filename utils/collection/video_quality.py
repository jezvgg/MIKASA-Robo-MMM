"""Decode every exported RGB frame and compare against its source render (E4)."""

from collections import defaultdict
from contextlib import ExitStack
import hashlib
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow.parquet as pq

from .contract import CAMERAS, read_json, write_json


def verify_video_quality(output, limit=2.0):
    root = Path(output)
    info = read_json(root / "meta/info.json")
    source = read_json(root / "source_h5_metadata.json")
    mapping = {e["episode_index"]: e for e in source["episodes"]}
    groups = defaultdict(list)
    for p in sorted((root / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(p).to_pylist():
            for camera in CAMERAS:
                key = f"observation.images.{camera}"
                prefix = f"videos/{key}/"
                path = root / info["video_path"].format(
                    video_key=key,
                    chunk_index=row[prefix + "chunk_index"],
                    file_index=row[prefix + "file_index"],
                )
                start = round(row[prefix + "from_timestamp"] * info["fps"])
                groups[path].append(
                    (start, row["length"], row["episode_index"], camera)
                )
    checks = []
    for path, segments in groups.items():
        segments.sort()
        with ExitStack() as stack:
            files = {
                eid: stack.enter_context(h5py.File(mapping[eid]["source_h5"], "r"))
                for _, _, eid, _ in segments
                if Path(mapping[eid]["source_h5"]).is_file()
            }
            by_frame = {}
            stats = {}
            digests = {}
            references = {}
            for start, length, eid, camera in segments:
                if eid not in files:
                    reference = mapping[eid].get("render_verification", {}).get(camera)
                    if (
                        not reference
                        or reference.get("frames") != length
                        or reference.get("seed") != mapping[eid]["scene_seed"]
                        or reference.get("camera") != camera
                        or len(reference.get("decoded_rgb_sha256", "")) != 64
                        or not 0
                        <= reference.get("mean_absolute_error_255", -1)
                        <= limit
                    ):
                        raise ValueError(
                            "Missing full-render evidence for retained RGB"
                        )
                    references[eid, camera] = reference
                digests[eid, camera] = hashlib.sha256()
                stats[(eid, camera)] = dict(
                    episode_index=eid,
                    seed=mapping[eid]["scene_seed"],
                    camera=camera,
                    frames=0,
                    pixel_error_sum=0.0,
                    pixels=0,
                    max_frame_mae=0.0,
                    verification=(
                        "source_render"
                        if eid in files
                        else "render_verified_rgb_digest"
                    ),
                )
                for i in range(length):
                    if start + i in by_frame:
                        raise ValueError("Overlapping episode video timestamps")
                    by_frame[start + i] = (eid, camera, i)
            covered = set(by_frame)
            video = stack.enter_context(av.open(str(path)))
            for frame in video.decode(video=0):
                index = round(float(frame.pts * frame.time_base) * info["fps"])
                if index not in by_frame:
                    if index in covered:
                        raise ValueError("Duplicate decoded video timestamp")
                    continue
                eid, camera, t = by_frame.pop(index)
                actual = frame.to_ndarray(format="rgb24")
                if list(actual.shape) != list(CAMERAS[camera]):
                    raise ValueError("Decoded RGB shape differs from source")
                row = stats[(eid, camera)]
                if row["frames"] != t:
                    raise ValueError("Decoded frames are not in episode order")
                digests[eid, camera].update(actual.tobytes())
                row["frames"] += 1
                if eid in files:
                    expected = files[eid][f"traj_0/obs/sensor_data/{camera}/rgb"][2 * t]
                    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
                    row["pixels"] += delta.size
                    row["pixel_error_sum"] += float(delta.sum())
                    row["max_frame_mae"] = max(
                        row["max_frame_mae"], float(delta.mean())
                    )
            if by_frame:
                raise ValueError("Video missing required episode frames")
            for key, row in stats.items():
                row["decoded_rgb_sha256"] = digests[key].hexdigest()
                if key in references:
                    reference = references[key]
                    if row["decoded_rgb_sha256"] != reference["decoded_rgb_sha256"]:
                        raise ValueError(
                            "Decoded RGB differs from the render-verified recording"
                        )
                    row["mean_absolute_error_255"] = reference[
                        "mean_absolute_error_255"
                    ]
                    row["max_frame_mae"] = reference["max_frame_mae"]
                    row.pop("pixel_error_sum")
                    row.pop("pixels")
                else:
                    row["mean_absolute_error_255"] = row.pop(
                        "pixel_error_sum"
                    ) / row.pop("pixels")
                encoding = (
                    mapping[row["episode_index"]]
                    .get("video_encoding", {})
                    .get(row["camera"], {})
                )
                if "decoded_rgb_sha256" in encoding:
                    if (
                        encoding["decoded_rgb_sha256"] != row["decoded_rgb_sha256"]
                        or encoding.get("frames") != row["frames"]
                    ):
                        raise ValueError(
                            "RGB differs from the encoder-verified frame sequence"
                        )
                checks.append(row)
    passed = all(row["mean_absolute_error_255"] <= limit for row in checks)
    report = dict(
        status="success" if passed else "failed",
        limit_255=limit,
        scope="Every exported frame; mean per episode and camera",
        total_camera_frames=sum(row["frames"] for row in checks),
        results=checks,
    )
    write_json(root / "video_quality.json", report)
    if not passed:
        raise ValueError(
            "Encoded RGB exceeds checklist E4 limit; see video_quality.json"
        )
    return report
