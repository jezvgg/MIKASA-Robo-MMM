"""Merge verified batches while preserving every original campaign and attempt."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from .contract import read_json, write_json
from .export_lerobot import verify
from .source_storage import file_sha256
from .provenance import dataset_provenance_complete, export_provenance


def merge_provenance(metadata_list):
    if not metadata_list:
        raise ValueError("At least one source batch is required")
    first = metadata_list[0]
    merged = copy.deepcopy(first)
    runs, outcomes, episodes, seen_seeds, conversions = {}, {}, [], set(), {}
    for metadata in metadata_list:
        for key in ("control_hz", "policy_hz", "action_repeat", "tokenizer_sha256"):
            if metadata[key] != first[key]:
                raise ValueError(f"Incompatible {key}")
        source_runs = metadata.get(
            "source_runs",
            [dict(source_root=metadata["source_root"], run=metadata["source_run"])],
        )
        for source in source_runs:
            root, run = source["source_root"], source["run"]
            if run["signature"] != first["source_run"]["signature"]:
                raise ValueError("Cannot merge different task/robot/runtime signatures")
            if root in runs and runs[root] != run:
                raise ValueError("A campaign changed between source batches")
            runs[root] = copy.deepcopy(run)
        for item in metadata["source_episode_outcomes"]:
            row = copy.deepcopy(item)
            row.setdefault("source_root", metadata["source_root"])
            key = (row["source_root"], row["scene_seed"])
            if key in outcomes and outcomes[key] != row:
                raise ValueError(
                    "Campaign outcomes changed; refresh batch provenance first"
                )
            outcomes[key] = row
        input_conversions = metadata.get("conversion_runs")
        if input_conversions is None:
            conversion = dict(
                provenance=metadata.get("provenance", {}),
                exporter=metadata.get("exporter", {}),
                video=metadata.get("video", {}),
            )
            conversion["id"] = hashlib.sha256(
                json.dumps(conversion, sort_keys=True).encode()
            ).hexdigest()
            input_conversions = [conversion]
        input_ids = {c["id"] for c in input_conversions}
        if not input_ids or len(input_ids) != len(input_conversions):
            raise ValueError("Missing or duplicate conversion provenance")
        for conversion in input_conversions:
            key = conversion["id"]
            if key in conversions and conversions[key] != conversion:
                raise ValueError("Conversion provenance changed between batches")
            conversions[key] = copy.deepcopy(conversion)
        for item in metadata["episodes"]:
            if item.get("success") is not True or item.get("truncated") is not False:
                raise ValueError(
                    "Only successful, non-truncated episodes can be merged"
                )
            if item["scene_seed"] in seen_seeds:
                raise ValueError("Duplicate episode seed across batches")
            seen_seeds.add(item["scene_seed"])
            row = copy.deepcopy(item)
            row.update(episode_index=len(episodes))
            row.setdefault("source_root", metadata["source_root"])
            if "conversion_runs" not in metadata:
                row["conversion_id"] = input_conversions[0]["id"]
            if row.get("conversion_id") not in input_ids:
                raise ValueError("Episode has no original converter provenance")
            episodes.append(row)
    candidate_seeds = [seed for run in runs.values() for seed in run["seeds"]]
    if len(candidate_seeds) != len(set(candidate_seeds)):
        raise ValueError("Independent campaigns contain overlapping candidate seeds")
    expected_keys = [
        (root, seed) for root, run in runs.items() for seed in run["seeds"]
    ]
    if set(outcomes) != set(expected_keys):
        raise ValueError("Missing or unexpected source-attempt outcomes")
    if len({"numeric_source_h5" in e for e in episodes}) > 1:
        raise ValueError(
            "Prepare retention for every input batch before mixing storage modes"
        )
    merged["source_runs"] = [
        dict(source_root=root, run=run) for root, run in runs.items()
    ]
    merged["source_episode_outcomes"] = [outcomes[key] for key in expected_keys]
    merged["episodes"] = episodes
    merged["conversion_runs"] = list(conversions.values())
    # source_run remains the actual first campaign, never a fabricated union.
    # source_runs and per-episode roots identify all other real campaigns.
    return merged


def validate_release(metadata, output):
    if len(metadata["episodes"]) < 1000:
        if "test" not in Path(output).name.lower():
            raise ValueError("Label a sub-1000-episode merged dataset with -test-Nep")
    elif not dataset_provenance_complete(metadata):
        raise ValueError(
            "Checklist A2: commit and bind the merger and every input batch converter"
        )


def merge(roots, output, repo_id, *, link_videos=False):
    from lerobot.datasets.aggregate import aggregate_datasets
    from .dataset_metadata import write_dataset_metadata

    roots, output = [Path(p).resolve() for p in roots], Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    metadata_list = [read_json(p / "source_h5_metadata.json") for p in roots]
    metadata = merge_provenance(metadata_list)
    metadata["provenance"] = export_provenance(
        Path(__file__).resolve().parents[2], metadata["source_run"]["signature"]
    )
    validate_release(metadata, output)
    batches = []
    for root in roots:
        verify(root)
        batches.append(
            dict(
                root=str(root),
                metadata_sha256=file_sha256(root / "source_h5_metadata.json"),
                readback_sha256=file_sha256(root / "readback.json"),
                video_quality_sha256=file_sha256(root / "video_quality.json"),
            )
        )
    if link_videos:
        from .linked_aggregate import aggregate_linked, verify_links

        storage = aggregate_linked(
            [m["repository_id"] for m in metadata_list], repo_id, roots, output
        )
    else:
        aggregate_datasets(
            [m["repository_id"] for m in metadata_list],
            repo_id,
            roots=roots,
            aggr_root=output,
        )
        storage = dict(mode="copy_videos")
    metadata["repository_id"] = repo_id
    metadata["aggregation"] = dict(
        module=__name__,
        source_sha256=file_sha256(__file__),
        batches=batches,
        storage=storage,
    )
    write_json(output / "source_h5_metadata.json", metadata)
    write_dataset_metadata(output, metadata)
    verify(output)
    if link_videos:
        verify_links(storage["videos"])
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--link-videos",
        action="store_true",
        help="Hardlink immutable videos on the same filesystem; no video copying",
    )
    args = parser.parse_args()
    metadata = merge(
        args.inputs, args.output, args.repo_id, link_videos=args.link_videos
    )
    print(
        "Merged",
        len(metadata["episodes"]),
        "episodes from",
        len(metadata["source_runs"]),
        "campaigns",
    )


if __name__ == "__main__":
    main()
