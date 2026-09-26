"""Retain exact numerical recordings and verified RGB before pruning render copies.

The original 20 Hz trajectories, physical replays, JSON metadata and LeRobot
videos are never deleted. Only explicitly listed successful ``rgb/*/trajectory.h5``
files can be pruned, after a complete numerical and decoded-video verification.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from .contract import read_json, write_json
from .export_lerobot import verify
from .source_storage import file_sha256, numerical_source


def compact_recording(source, destination):
    """Copy every non-image field without resampling or changing its dtype."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".h5.tmp")
    with h5py.File(source) as src, h5py.File(temporary, "w") as dst:
        def copy_group(before, after):
            for key, value in before.attrs.items():
                after.attrs[key] = value
            for name, value in before.items():
                if value.name == "/traj_0/obs/sensor_data":
                    continue
                if isinstance(value, h5py.Group):
                    copy_group(value, after.create_group(name))
                else:
                    before.copy(value, after, name=name)
                    # HDF5 copy should be exact, but verify the actual payload.
                    if value.shape == ():
                        np.testing.assert_array_equal(value[()], after[name][()])
                    else:
                        for start in range(0, len(value), 512):
                            np.testing.assert_array_equal(value[start:start + 512],
                                                          after[name][start:start + 512])
        copy_group(src, dst)
    temporary.replace(destination)
    metadata = read_json(source.with_suffix(".json"))
    metadata["retained_payload"] = "Exact numerical fields and env_states; RGB in verified LeRobot videos"
    write_json(destination.with_suffix(".json"), metadata)


def prepare(root):
    from .dataset_metadata import write_dataset_metadata

    root = Path(root).resolve()
    # This also refreshes E4 with a digest over every decoded RGB pixel.
    verify(root)
    metadata_path = root / "source_h5_metadata.json"
    metadata = read_json(metadata_path)
    pending_path = root / "retention.prepare.json"
    if pending_path.exists():
        pending = read_json(pending_path)
        if (file_sha256(metadata_path) != pending["source_metadata_sha256"]
                and metadata != pending["metadata"]):
            raise ValueError("Dataset metadata changed during retention preparation")
        metadata = pending["metadata"]
    else:
        if (root / "numeric_sources").exists():
            raise FileExistsError("Unclaimed numeric_sources directory")
        pending = dict(source_metadata_sha256=file_sha256(metadata_path), metadata=metadata)
        write_json(pending_path, pending)
    quality = read_json(root / "video_quality.json")
    by_episode = {}
    for row in quality["results"]:
        by_episode.setdefault(row["episode_index"], {})[row["camera"]] = row
    records = []
    for episode in metadata["episodes"]:
        source = Path(episode["source_h5"])
        if "numeric_source_h5" not in episode:
            destination = root / "numeric_sources" / f"{episode['scene_seed']}.h5"
            # The pending manifest reserves this generated location. Rebuilding
            # an interrupted copy is safe because the source RGB still exists.
            compact_recording(source, destination)
            episode.update(numeric_source_h5=str(destination),
                           numeric_source_sha256=file_sha256(destination),
                           numeric_metadata_sha256=file_sha256(destination.with_suffix(".json")),
                           render_source_sha256=file_sha256(source),
                           render_source_bytes=source.stat().st_size,
                           render_verification=by_episode[episode["episode_index"]])
        numerical_source(episode)
        records.append({key: episode[key] for key in (
            "scene_seed", "source_h5", "render_source_sha256", "render_source_bytes",
            "numeric_source_h5", "numeric_source_sha256", "numeric_metadata_sha256")})
        # Commit each finished copy to the manifest for restart after interruption.
        write_json(pending_path, pending)
    write_dataset_metadata(root, metadata)
    write_json(metadata_path, metadata)
    report = dict(version=1, status="prepared", episodes=records,
                  preserved="oracle/native/validated trajectories, every JSON, numeric sources and LeRobot videos",
                  render_verification="Every decoded RGB frame matched the source render; subsequent checks require identical decoded-pixel digests")
    write_json(root / "retention.json", report)
    verify(root)
    return report


def prune(root):
    root = Path(root).resolve()
    report = read_json(root / "retention.json")
    metadata = read_json(root / "source_h5_metadata.json")
    verify(root)
    candidates = []
    for episode in metadata["episodes"]:
        numerical_source(episode)
        source = Path(episode["source_h5"])
        source_root = Path(episode.get("source_root", metadata["source_root"]))
        expected = source_root / "rgb" / str(episode["scene_seed"]) / "trajectory.h5"
        if source.absolute() != expected.absolute() or source.is_symlink():
            raise ValueError("Pruning is restricted to explicit campaign RGB copies")
        result = read_json(source.with_name("result.json"))
        if result.get("success") is not True:
            raise ValueError("Cannot prune unfinished or unsuccessful RGB")
        if source.exists() and file_sha256(source) != episode["render_source_sha256"]:
            raise ValueError("RGB recording changed since retention verification")
        candidates.append(source)
    # Validate all candidates before the first deletion; keep a restartable log.
    report.update(status="pruning", deleted=report.get("deleted", []))
    write_json(root / "retention.json", report)
    for source in candidates:
        if source.exists():
            source.unlink()
            report["deleted"].append(str(source))
            write_json(root / "retention.json", report)
    verify(root)
    report["status"] = "pruned_and_verified"
    write_json(root / "retention.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "prune"))
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    result = prepare(args.dataset) if args.command == "prepare" else prune(args.dataset)
    print(result["status"], len(result["episodes"]), "episodes")


if __name__ == "__main__":
    main()
