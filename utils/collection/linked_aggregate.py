"""Merge LeRobot shards by linking immutable videos and rewriting only metadata/data.

Every source video keeps its own destination file and original timestamps. No
concatenation, re-encoding, or source write occurs. Hardlinks require one filesystem;
there is deliberately no silent copy fallback that could exceed the storage budget.
"""

import os
from pathlib import Path
import stat


def file_index(number, chunk_size):
    if number < 0 or chunk_size < 1:
        raise ValueError("Nonnegative file number and positive chunk size required")
    return divmod(number, chunk_size)


def remap_pairs(frame, prefix, mapping):
    """Map each original file independently, including files shared by episodes."""
    chunk, file = f"{prefix}/chunk_index", f"{prefix}/file_index"
    pairs = [mapping[(int(c), int(f))] for c, f in zip(frame[chunk], frame[file])]
    frame[chunk] = [p[0] for p in pairs]
    frame[file] = [p[1] for p in pairs]


def link_video(source, destination, root):
    source, destination, root = Path(source), Path(destination), Path(root).resolve()
    if source.is_symlink() or not source.resolve().is_relative_to(root):
        raise ValueError(
            "Source video must be a regular file inside its source dataset"
        )
    before = source.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise ValueError("Source video is missing, empty, or not a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination, follow_symlinks=False)
    after, linked = source.stat(), destination.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if identity(before) != identity(after) or identity(before) != identity(linked):
        raise RuntimeError("Source video changed while creating its hardlink")
    return dict(
        source=str(source),
        destination=str(destination),
        size_bytes=before.st_size,
        device=before.st_dev,
        inode=before.st_ino,
        mtime_ns=before.st_mtime_ns,
    )


def verify_links(records):
    for row in records:
        expected = (row["device"], row["inode"], row["size_bytes"], row["mtime_ns"])
        for key in ("source", "destination"):
            p = Path(row[key])
            s = p.stat()
            if (
                p.is_symlink()
                or (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) != expected
            ):
                raise ValueError(f"Video file changed during aggregation: {p}")


def aggregate_linked(repo_ids, repo_id, roots, output, *, chunk_size=1000):
    """Use upstream schemas/statistics with explicit source-file index mappings."""
    import pandas as pd
    from lerobot.datasets.aggregate import (
        finalize_aggregation,
        update_data_df,
        validate_all_metadata,
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import (
        DEFAULT_DATA_PATH,
        DEFAULT_EPISODES_PATH,
        DEFAULT_VIDEO_PATH,
    )

    roots, output = [Path(p).resolve() for p in roots], Path(output).resolve()
    if not roots or len(roots) != len(repo_ids) or chunk_size < 1:
        raise ValueError(
            "Each nonempty source needs a repository id; chunk size must be positive"
        )
    if output.exists():
        raise FileExistsError(output)
    if any(output.is_relative_to(root) for root in roots):
        raise ValueError("Output cannot be inside a source dataset")
    all_meta = [LeRobotDatasetMetadata(i, root=r) for i, r in zip(repo_ids, roots)]
    fps, robot_type, features = validate_all_metadata(all_meta)
    if any(f["dtype"] == "image" for f in features.values()):
        raise ValueError("Linked merger supports video datasets, not embedded images")
    video_keys = [k for k, f in features.items() if f["dtype"] == "video"]
    dst = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        root=output,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=bool(video_keys),
        chunks_size=chunk_size,
    )
    tasks = pd.concat([m.tasks for m in all_meta]).index.unique()
    dst.tasks = pd.DataFrame({"task_index": range(len(tasks))}, index=tasks)
    records = []
    data_number = 0
    video_numbers = {key: 0 for key in video_keys}
    for meta_number, src in enumerate(all_meta):
        episodes = src.episodes.to_pandas()
        if sorted(episodes["episode_index"].tolist()) != list(
            range(src.total_episodes)
        ):
            raise ValueError("Source episode indices are not contiguous")
        for key in video_keys:
            prefix = f"videos/{key}"
            mapping = {}
            for c, f in sorted(
                set(
                    zip(
                        episodes[f"{prefix}/chunk_index"],
                        episodes[f"{prefix}/file_index"],
                    )
                )
            ):
                old = (int(c), int(f))
                new = file_index(video_numbers[key], chunk_size)
                video_numbers[key] += 1
                source = src.root / src.info["video_path"].format(
                    video_key=key, chunk_index=old[0], file_index=old[1]
                )
                dest = output / DEFAULT_VIDEO_PATH.format(
                    video_key=key, chunk_index=new[0], file_index=new[1]
                )
                records.append(link_video(source, dest, src.root))
                mapping[old] = new
            remap_pairs(episodes, prefix, mapping)
            # Preserve from/to timestamps: each original video remains separate.
        mapping = {}
        for c, f in sorted(
            set(zip(episodes["data/chunk_index"], episodes["data/file_index"]))
        ):
            old = (int(c), int(f))
            new = file_index(data_number, chunk_size)
            data_number += 1
            source = src.root / src.info["data_path"].format(
                chunk_index=old[0], file_index=old[1]
            )
            if source.is_symlink() or not source.resolve().is_relative_to(src.root):
                raise ValueError("Source parquet must stay inside its dataset")
            frame = update_data_df(pd.read_parquet(source), src, dst)
            dest = output / DEFAULT_DATA_PATH.format(
                chunk_index=new[0], file_index=new[1]
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(dest, index=False)
            mapping[old] = new
        remap_pairs(episodes, "data", mapping)
        episodes["episode_index"] += dst.info["total_episodes"]
        for key in ["dataset_from_index", "dataset_to_index"]:
            episodes[key] += dst.info["total_frames"]
        mc, mf = file_index(meta_number, chunk_size)
        episodes["meta/episodes/chunk_index"] = mc
        episodes["meta/episodes/file_index"] = mf
        dest = output / DEFAULT_EPISODES_PATH.format(chunk_index=mc, file_index=mf)
        dest.parent.mkdir(parents=True, exist_ok=True)
        episodes.to_parquet(dest, index=False)
        dst.info["total_episodes"] += src.total_episodes
        dst.info["total_frames"] += src.total_frames
    verify_links(records)
    finalize_aggregation(dst, all_meta)
    return dict(
        mode="hardlink_immutable_videos",
        additional_video_payload_bytes=0,
        linked_video_bytes=sum(r["size_bytes"] for r in records),
        videos=records,
        data_files=data_number,
        metadata_files=len(all_meta),
        note="Do not edit linked videos in place. Removing an input path preserves the output link. Numerical data and metadata are independent rewritten files.",
    )
