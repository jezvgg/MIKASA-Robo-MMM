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


def episode_stats(row, template):
    """One episode's stored statistics (all features, quantiles included) in the
    shapes of the dataset-level statistics, for LeRobot's `aggregate_stats`."""
    import numpy as np

    out = {}
    for feature, stats in template.items():
        out[feature] = {}
        for name, reference in stats.items():
            column = f"stats/{feature}/{name}"
            if column not in row:
                raise ValueError(f"G4: episode statistics lack {column}")
            value = row[column]
            array = np.array(value.tolist() if hasattr(value, "tolist") else value,
                             dtype=np.float64)
            out[feature][name] = array.reshape(np.shape(reference))
    return out


def aggregate_linked(repo_ids, repo_id, roots, output, *, chunk_size=1000, keep_seeds=None):
    """Use upstream schemas/statistics with explicit source-file index mappings.

    `keep_seeds` (optional) keeps only episodes whose `episode_seed` is listed:
    their rows are copied with contiguous new indices, videos are still linked
    whole (kept episodes keep their original timestamps; frames of dropped
    episodes stay unreferenced in the linked files) and the dataset statistics
    are aggregated from the kept episodes' own statistics.
    """
    import pandas as pd
    from lerobot.datasets.aggregate import (
        finalize_aggregation,
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
    kept_stats = []
    keep = None if keep_seeds is None else {int(x) for x in keep_seeds}
    for meta_number, src in enumerate(all_meta):
        # The runtime src.episodes view deliberately removes every stats/ column.
        # Preserve quantiles and all other original episode metadata for G4.
        from .normalization_stats import read_episode_metadata

        episodes = read_episode_metadata(src.root)
        if sorted(episodes["episode_index"].tolist()) != list(
            range(src.total_episodes)
        ):
            raise ValueError("Source episode indices are not contiguous")
        if keep is not None:
            episodes = episodes[episodes["episode_seed"].astype(int).isin(keep)]
            episodes = episodes.sort_values("episode_index", ignore_index=True)
            if not len(episodes):
                raise ValueError("A source contributes no selected episode; omit it")
        # New contiguous episode numbers and frame ranges in the destination.
        old_episodes = episodes["episode_index"].astype(int).tolist()
        renumber = {old: dst.info["total_episodes"] + k for k, old in enumerate(old_episodes)}
        lengths = (episodes["dataset_to_index"] - episodes["dataset_from_index"]).astype(int).tolist()
        starts = [dst.info["total_frames"] + sum(lengths[:k]) for k in range(len(lengths))]
        new_start = dict(zip(old_episodes, starts))
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
            frame = pd.read_parquet(source)
            frame = frame[frame["episode_index"].astype(int).isin(renumber)].copy()
            old_index = frame["episode_index"].astype(int)
            frame["index"] = [new_start[e] + int(f) for e, f in zip(old_index, frame["frame_index"])]
            frame["episode_index"] = old_index.map(renumber).to_numpy()
            src_task_names = src.tasks.index.take(frame["task_index"].to_numpy())
            frame["task_index"] = dst.tasks.loc[src_task_names, "task_index"].to_numpy()
            dest = output / DEFAULT_DATA_PATH.format(
                chunk_index=new[0], file_index=new[1]
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(dest, index=False)
            mapping[old] = new
        remap_pairs(episodes, "data", mapping)
        episodes["episode_index"] = [renumber[e] for e in old_episodes]
        episodes["dataset_from_index"] = starts
        episodes["dataset_to_index"] = [a + n for a, n in zip(starts, lengths)]
        kept_stats += [episode_stats(row, src.stats) for _, row in episodes.iterrows()]
        mc, mf = file_index(meta_number, chunk_size)
        episodes["meta/episodes/chunk_index"] = mc
        episodes["meta/episodes/file_index"] = mf
        dest = output / DEFAULT_EPISODES_PATH.format(chunk_index=mc, file_index=mf)
        dest.parent.mkdir(parents=True, exist_ok=True)
        episodes.to_parquet(dest, index=False)
        dst.info["total_episodes"] += len(old_episodes)
        dst.info["total_frames"] += sum(lengths)
    verify_links(records)
    if keep is None:
        finalize_aggregation(dst, all_meta)
    else:
        from lerobot.datasets.compute_stats import aggregate_stats
        from lerobot.datasets.utils import write_info, write_stats, write_tasks

        write_tasks(dst.tasks, dst.root)
        dst.info.update(total_tasks=len(dst.tasks), splits={"train": f"0:{dst.info['total_episodes']}"})
        write_info(dst.info, dst.root)
        dst.stats = aggregate_stats(kept_stats)
        write_stats(dst.stats, dst.root)
    return dict(
        mode="hardlink_immutable_videos",
        additional_video_payload_bytes=0,
        linked_video_bytes=sum(r["size_bytes"] for r in records),
        videos=records,
        data_files=data_number,
        metadata_files=len(all_meta),
        selected_episodes=None if keep is None else dst.info["total_episodes"],
        note="Do not edit linked videos in place. Removing an input path preserves the output link. Numerical data and metadata are independent rewritten files.",
    )
