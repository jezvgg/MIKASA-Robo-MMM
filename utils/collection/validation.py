"""Freeze exactly 100 independent, planner-success CabinetSearch validation seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

from .contract import read_json, recording_info, write_json


def select_validation(candidates, excluded, count=100):
    """Select by scene seed, never by replay/model performance or worker speed."""
    seen = set()
    successes = []
    for candidate in sorted(candidates, key=lambda item: item["seed"]):
        seed = candidate["seed"]
        if seed in seen:
            raise ValueError(f"Duplicate scene seed across candidate pools: {seed}")
        seen.add(seed)
        if seed in excluded:
            raise ValueError(f"Validation candidate overlaps training/development: {seed}")
        if candidate["planner_success"]:
            successes.append(candidate)
    if len(successes) < count:
        raise ValueError(f"Only {len(successes)} planner successes; need {count}")
    return successes[:count]


def freeze(roots, excluded_roots, output, extra_excluded=()):
    if output.exists():
        raise FileExistsError("Validation lists are immutable; do not replace after policy failures")
    runs = [read_json(root / "run.json") for root in roots]
    signature = runs[0]["signature"]
    if any(run["purpose"] != "validation" or run["signature"] != signature for run in runs):
        raise ValueError("Validation sources must share one frozen implementation and purpose")
    excluded = set(extra_excluded)
    exclusions = []
    for root in excluded_roots:
        run = read_json(root / "run.json")
        if run["purpose"] not in {"train", "development"}:
            raise ValueError("An exclusion pool must be training or development")
        # Exclude the entire assigned pool, including interrupted/failed attempts.
        excluded.update(run["seeds"])
        exclusions.append(dict(run=str(root), purpose=run["purpose"], seeds=run["seeds"]))
    candidates = []
    for root, run in zip(roots, runs):
        for seed in run["seeds"]:
            directory = root / "oracle" / str(seed)
            if not (directory / "result.json").exists():
                raise ValueError(f"Incomplete candidate pool: {root}, seed {seed}")
            result = read_json(directory / "result.json")
            success = result["status"] == "success"
            if success:
                recording_info(directory / "trajectory.h5", stage="oracle")
                if not result.get("task_metrics", {}).get("success", False):
                    raise ValueError("Planner success is not backed by physical task metrics")
            replay_path = root / "validated" / str(seed) / "result.json"
            replay = read_json(replay_path) if replay_path.exists() else {"status": "not_run"}
            candidates.append(dict(seed=seed, planner_success=success,
                source=str(directory / "result.json"), planner_result=result,
                ten_hz_replay=replay))
    selected = select_validation(candidates, excluded, count=100)
    manifest = dict(version=1, task=signature["profile"]["env_id"],
        selection="first_100_successful_scene_seeds_in_ascending_order",
        selection_requires="motion_planner_success_true",
        ten_hz_replay_is_selection_condition=False, immutable_after_policy_evaluation=True,
        seeds=[item["seed"] for item in selected], selected=selected,
        candidate_attempts=candidates, exclusions=exclusions,
        extra_excluded_seeds=sorted(extra_excluded), signature=signature)
    write_json(output, manifest)
    print(f"Frozen {len(selected)} validation seeds from {len(candidates)} candidates: {output}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--exclude-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--exclude-seeds", nargs="*", type=int, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze([p.resolve() for p in args.runs], [p.resolve() for p in args.exclude_runs],
           args.output.resolve(), args.exclude_seeds)


if __name__ == "__main__":
    main()
