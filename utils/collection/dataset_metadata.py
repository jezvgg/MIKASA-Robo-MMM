"""Checklist G5 metadata, derived from the exact exported H5 episodes.

This only enriches episode metadata. It cannot certify motion quality or turn old
trajectories into compliant demonstrations.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .contract import episode_summary, read_json, write_json
from .source_storage import numerical_source


def episode_rows(metadata):
    rows = {}
    seeds = set()
    for item in metadata["episodes"]:
        index = item["episode_index"]
        if index in rows or item["scene_seed"] in seeds:
            raise ValueError("Duplicate episode index or source seed")
        seeds.add(item["scene_seed"])
        path = numerical_source(item)
        source = read_json(path.with_suffix(".json"))
        if source["episodes"][0]["episode_seed"] != item["scene_seed"]:
            raise ValueError("Episode seed differs from source recording")
        with h5py.File(path, "r") as h5:
            traj = h5["traj_0"]
            summary = episode_summary(traj)
            for key, value in summary.items():
                if item[key] != value:
                    raise ValueError(f"Exported {key} differs from H5")
            if len(traj["actions"]) != item["frames"] * 2:
                raise ValueError("Episode frame count disagrees with held-action H5")
            for key in ["terminated", "truncated"]:
                if item[key] != bool(traj[key][-1]):
                    raise ValueError(f"Exported {key} differs from H5")
        rows[index] = dict(
            episode_seed=item["scene_seed"],
            planner_seed=item["planner_seed"],
            waypoint_noise_seed=item["waypoint_noise_seed"],
            episode_length=item["frames"],
            episode_duration_s=item["duration_seconds"],
            reward_sum=item["reward_sum"],
            success=item["success"],
            success_once=item["success_once"],
            terminated=item["terminated"],
            truncated=item["truncated"],
            source_h5=item["source_h5"],
            source_runtime_sha256=item["source_sha256"],
        )
        for key in (
            "numeric_source_h5",
            "numeric_source_sha256",
            "numeric_metadata_sha256",
        ):
            if key in item:
                rows[index][key] = item[key]
    return rows


def write_dataset_metadata(output, metadata):
    output = Path(output)
    rows = episode_rows(metadata)
    files = sorted((output / "meta/episodes").rglob("*.parquet"))
    seen, pending = set(), []
    for path in files:
        table = pq.read_table(path)
        ids = table["episode_index"].to_pylist()
        if any(i not in rows or i in seen for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Unexpected or repeated episode metadata row")
        seen.update(ids)
        for i, length in zip(ids, table["length"].to_pylist()):
            if length != rows[i]["episode_length"]:
                raise ValueError("LeRobot episode length differs from H5 mapping")
        for key in next(iter(rows.values())):
            values = [rows[i][key] for i in ids]
            if key in table.column_names:
                if table[key].to_pylist() != values:
                    raise ValueError(f"Refusing to replace conflicting {key}")
            else:
                table = table.append_column(key, pa.array(values))
        pending.append((path, table))
    if seen != set(rows):
        raise ValueError("Missing LeRobot episode rows")
    ordered = [rows[i] for i in sorted(rows)]
    outcomes = metadata.get("source_episode_outcomes", [])
    attempted = [r for r in outcomes if r.get("status") != "not_run"]
    successes = sum(r.get("success") is True for r in attempted)
    report = {
        "schema_version": 1,
        "source_format": "MIKASA HDF5 (not converted from RLDS)",
        "episode_indices": sorted(rows),
        "episode_seeds": [r["episode_seed"] for r in ordered],
        "episode_lengths": [r["episode_length"] for r in ordered],
        "episode_durations_s": [r["episode_duration_s"] for r in ordered],
        "success": [r["success"] for r in ordered],
        "success_once": [r["success_once"] for r in ordered],
        "reward_sums": [r["reward_sum"] for r in ordered],
        "terminated": [r["terminated"] for r in ordered],
        "truncated": [r["truncated"] for r in ordered],
        "episodes": ordered,
        "all_collection_attempts": outcomes,
        "planner_success_rate": successes / len(attempted) if attempted else None,
        "planner_successes": successes,
        "planner_attempts": len(attempted),
        "failed_seeds": [
            r["scene_seed"] for r in attempted if r.get("success") is not True
        ],
        "control_hz": metadata["control_hz"],
        "dataset_hz": metadata["policy_hz"],
        "resampling": metadata["resampling"],
        "profile": metadata["source_run"]["signature"]["profile"],
        "source_runtime_sha256": metadata["source_run"]["signature"]["code_sha256"],
        "original_exporter": metadata["exporter"],
        "metadata_writer_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "qualification": "Format metadata only; see full DATASET_CHECKLIST audit",
    }
    for path, table in pending:
        temp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, temp)
        temp.replace(path)
    write_json(output / "meta/source_rlds_metadata.json", report)
    verify_dataset_metadata(output, metadata)


def verify_dataset_metadata(output, metadata):
    from .normalization_stats import verify_normalization

    output = Path(output)
    verify_normalization(output)
    expected = episode_rows(metadata)
    seen = set()
    for path in sorted((output / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            index = row["episode_index"]
            if index in seen or index not in expected:
                raise ValueError("Unexpected/repeated metadata episode index")
            seen.add(index)
            for key, value in expected[index].items():
                if row.get(key) != value:
                    raise ValueError(f"Episode {index} {key} differs from source H5")
    if seen != set(expected):
        raise ValueError("Missing metadata episodes")
    report = read_json(output / "meta/source_rlds_metadata.json")
    for field, key in [
        ("episode_seeds", "episode_seed"),
        ("episode_lengths", "episode_length"),
        ("episode_durations_s", "episode_duration_s"),
        ("success", "success"),
        ("success_once", "success_once"),
        ("reward_sums", "reward_sum"),
        ("terminated", "terminated"),
        ("truncated", "truncated"),
    ]:
        if report[field] != [expected[i][key] for i in sorted(expected)]:
            raise ValueError(f"Metadata summary {field} differs from H5")
