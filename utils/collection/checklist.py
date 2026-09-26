"""Read-only evidence for DATASET_CHECKLIST.md; unmeasured items remain UNKNOWN.

Run in the isolated LeRobot export environment. Never repair actions in a dataset:
planner changes need fresh physical rollouts and replay qualification.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path

import h5py
import numpy as np

ITEMS = [
    f"{section}{i}"
    for section, n in zip("ABCDEFGH", (5, 6, 3, 9, 5, 3, 5, 5))
    for i in range(1, n + 1)
]
CAMS = {
    "left_base_camera_link": [256, 256, 3],
    "right_base_camera_link": [256, 256, 3],
    "fetch_hand": [128, 128, 3],
}
ROLL = ["upperarm_roll_joint", "forearm_roll_joint", "wrist_roll_joint"]
ARM = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]


def read(path):
    return json.loads(Path(path).read_text())


def longest_true(values):
    best = run = 0
    for value in values:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def motion_metrics(actions, qpos, names, fps=10):
    a, q = np.asarray(actions), np.asarray(qpos)
    if a.ndim != 2 or a.shape[1] != 13 or not len(a):
        raise ValueError("Expected nonempty 13D actions")
    if q.ndim != 2 or q.shape[1] != 15 or len(q) < len(a):
        raise ValueError("Expected aligned 15D qpos")
    if not np.isfinite(a).all() or not np.isfinite(q).all():
        raise ValueError("Nonfinite actions or qpos")
    ridx = [names.index(n) for n in ROLL]
    rolls = q[:, ridx]
    equal = np.all(np.abs(np.diff(a, axis=0)) <= 1e-4, axis=1)
    directions = np.sign(a[np.abs(a[:, 11]) > 1e-6, 11])
    tiny = (np.abs(a[:, 11:13]) > 0) & (np.abs(a[:, 11:13]) <= 1e-12)
    # Segment by *command* sign, integrate actual qpos motion (not normalized action).
    # Root yaw is continuous in DSFetch; unwrapping hides genuine >pi jumps.
    yaw = q[:, names.index("root_z_rotation_joint")]
    dyaw = np.diff(yaw)
    segments = []
    sign = 0
    angle = 0.0
    for i, command in enumerate(a[:, 12]):
        new = int(np.sign(command)) if abs(command) > 1e-6 else 0
        if new != sign:
            if sign:
                segments.append(abs(angle))
            angle, sign = 0.0, new
        if new and i < len(dyaw):
            angle += float(dyaw[i])
    if sign:
        segments.append(abs(angle))
    changes = np.flatnonzero(np.diff(a[:, 7]) != 0) + 1
    close = changes[a[changes, 7] < a[changes - 1, 7]]
    combos = [
        [
            int(np.sign(q[i, names.index("elbow_flex_joint")])),
            int(np.sign(q[i, names.index("wrist_flex_joint")])),
        ]
        for i in close
    ]
    return {
        "frames": len(a),
        "gripper_values": np.unique(a[:, 7]).tolist(),
        "base_abs_max": float(np.abs(a[:, 11:13]).max()),
        "base_tiny_nonzero_count": int(tiny.sum()),
        "roll_min": rolls.min(axis=0).tolist(),
        "roll_max": rolls.max(axis=0).tolist(),
        "roll_span": np.ptp(rolls, axis=0).tolist(),
        "roll_command_min": a[:, [2, 4, 6]].min(axis=0).tolist(),
        "roll_command_max": a[:, [2, 4, 6]].max(axis=0).tolist(),
        "max_continuous_turn_rad": max(segments, default=0.0),
        "total_turn_rad": float(np.abs(dyaw).sum()),
        "reverse_frames": int(np.count_nonzero(a[:, 11] < -1e-6)),
        "base_direction_changes": (
            int(np.count_nonzero(np.diff(directions))) if len(directions) else 0
        ),
        "identical_action_fraction": float(equal.mean()) if len(equal) else 0.0,
        "longest_constant_action_frames": longest_true(equal) + 1,
        "longest_constant_action_seconds": (longest_true(equal) + 1) / fps,
        "close_sign_combinations": combos,
        "gripper_switch_steps": changes.tolist(),
        "initial_qpos": q[0].tolist(),
        "action_std": a.astype(np.float64).std(axis=0).tolist(),
    }


def pause_check(actions, profile=None):
    """Report literal D6 and the explicitly authorized initial cue exception."""
    a = np.asarray(actions)
    def metrics(values):
        equal = np.all(np.abs(np.diff(values, axis=0)) <= 1e-4, axis=1)
        fraction = float(equal.mean()) if len(equal) else 0.
        longest = longest_true(equal) + 1 if len(values) else 0
        return {"identical_action_fraction": fraction,
                "longest_constant_action_frames": longest,
                "passed": bool(len(values) and fraction <= .05 and longest <= 20)}
    raw = metrics(a)
    profile = profile or {}
    exception = profile.get("quality_exceptions", {}).get("D6", {})
    approved = (profile.get("env_id") == "MikasaSeasonDish-v0"
                and profile.get("profile_version", 0) >= 4
                and exception.get("initial_observation_control_steps") == 100
                and exception.get("phase") == "fridge_cue_observation"
                and exception.get("preserve_all_frames") is True)
    frames = 50 if approved else 0
    valid = len(a) > frames and (not approved or np.all(a[:frames, 11:13] == 0))
    remaining = metrics(a[frames:])
    return {"raw": raw, "excluded_initial_frames": frames,
            "exception": exception if approved else None,
            "after_observation": remaining,
            "passed": bool(valid and remaining["passed"])}


def audit(dataset, load_samples=False, tokenizer=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(dataset).resolve()
    result = {
        "dataset": str(root),
        "checks": {
            k: {"status": "UNKNOWN", "evidence": "Not measured by this audit"}
            for k in ITEMS
        },
    }
    checks = result["checks"]

    def setcheck(key, status, evidence):
        checks[key] = {"status": status, "evidence": evidence}

    def verdict(key, condition, evidence):
        setcheck(key, "PASS" if condition else "FAIL", evidence)

    info = read(root / "meta/info.json")
    files = sorted((root / "data").rglob("*.parquet"))
    result["info"] = {
        k: info.get(k)
        for k in (
            "total_episodes",
            "total_frames",
            "fps",
            "robot_type",
            "codebase_version",
        )
    }
    if not files:
        setcheck(
            "G1",
            "UNKNOWN",
            "Data parquet files absent: metadata-only historical dataset",
        )
        return result
    table = pa.concat_tables([pq.read_table(p) for p in files]).sort_by("index")
    episodes = pa.concat_tables(
        [pq.read_table(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))]
    )
    cols = episodes.column_names
    stats = read(root / "meta/stats.json")
    values = {
        k: np.asarray(table[k].to_pylist())
        for k in [
            "action",
            "observation.state",
            "global_state",
            "episode_index",
            "timestamp",
        ]
        if k in table.column_names
    }
    meta_path = root / "source_h5_metadata.json"
    metadata = read(meta_path) if meta_path.exists() else {}
    names = info["features"].get("observation.state", {}).get("names", [])
    if isinstance(names, dict):
        names = names.get("joints", [])
    if len(names) != 12:
        raise ValueError("Need exact native joint names; cannot infer order")
    names = ["root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"] + names
    per_episode, raw_checks, profiles = [], [], []
    mappings = {e["episode_index"]: e for e in metadata.get("episodes", [])}
    for eid in np.unique(values["episode_index"]):
        mask = values["episode_index"] == eid
        a = values["action"][mask]
        q = np.concatenate(
            [values["global_state"][mask], values["observation.state"][mask]], axis=1
        )
        row = motion_metrics(a, q, names, info["fps"])
        row["D6_pause_check"] = pause_check(a)
        row["episode_index"] = int(eid)
        mapping = mappings.get(int(eid), {})
        row["seed"] = mapping.get("scene_seed")
        from .source_storage import numerical_source
        hp = numerical_source(mapping) if "source_h5" in mapping else Path("/missing")
        if hp.is_file():
            hmeta = read(hp.with_suffix(".json"))["mikasa_data"]
            profiles.append(hmeta["profile"])
            row["D6_pause_check"] = pause_check(a, hmeta["profile"])
            source_root = Path(mapping.get("source_root", metadata["source_root"]))
            original = (
                source_root / "oracle" / str(mapping["scene_seed"]) / "trajectory.h5"
            )
            raw = {
                "episode_index": int(eid),
                "source": str(hp),
                "oracle": str(original),
            }
            with h5py.File(hp) as h5:
                t = h5["traj_0"]
                raw["final_success"] = bool(t["success"][-1])
                raw["truncated"] = bool(t["truncated"][-1])
                raw["control_steps"] = len(t["actions"])
                raw["horizon"] = hmeta.get("task_config", {}).get("horizon")
                raw["held_action_matches_export"] = np.array_equal(a, t["actions"][::2])
                raw["proprio_matches_export"] = np.array_equal(
                    q, t["obs/agent/qpos"][:][:-1:2]
                )
            if original.is_file():
                with h5py.File(original) as h5:
                    t = h5["traj_0"]
                    src = t["actions"][:]
                    switches = np.flatnonzero(np.diff(src[:, 7]) != 0) + 1
                    raw["gripper_switch_steps"] = switches.tolist()
                    raw["odd_gripper_switches"] = switches[switches % 2 == 1].tolist()
                    raw["original_steps"] = len(src)
                    raw["initial_qpos"] = t["qpos"][0].tolist()
            raw_checks.append(raw)
        row["timestamp_error"] = float(
            np.abs(values["timestamp"][mask] - np.arange(len(a)) / 10).max()
        )
        per_episode.append(row)
    result.update(episodes=per_episode, raw_checks=raw_checks)
    n = len(per_episode)
    all_raw = len(raw_checks) == n
    verdict(
        "B3",
        all(set(e["gripper_values"]) <= {-1.0, 1.0} for e in per_episode),
        "Exact unique values recorded per episode",
    )
    verdict(
        "B4",
        all(
            e["base_abs_max"] <= 1 and e["base_tiny_nonzero_count"] == 0
            for e in per_episode
        ),
        "All base entries checked; numerical zero threshold 1e-12",
    )
    verdict(
        "B5",
        all_raw
        and all(e["proprio_matches_export"] for e in raw_checks)
        and values["observation.state"].shape[1:] == (12,)
        and values["global_state"].shape[1:] == (3,)
        and "observation.global_state" not in table.column_names,
        "Every numerical row compared with source RGB H5 qpos slices",
    )
    state_std = values["observation.state"].astype(np.float64).std(axis=0)
    action_std = values["action"].astype(np.float64).std(axis=0)
    result["state_std"] = state_std.tolist()
    result["action_std"] = action_std.tolist()
    setcheck(
        "B6",
        "PARTIAL",
        "Per-column std measured; task-specific required motion and noise floor need review",
    )
    starts = (
        np.asarray(
            [
                e.get("initial_qpos", per_episode[i]["initial_qpos"])
                for i, e in enumerate(raw_checks)
            ]
        )
        if all_raw
        else np.asarray([e["initial_qpos"] for e in per_episode])
    )
    initial_std = starts.astype(np.float64).std(axis=0)
    result["initial_qpos_std"] = dict(zip(names, initial_std.tolist()))
    required = ARM + ["torso_lift_joint", "head_pan_joint", "head_tilt_joint"]
    varied = all(initial_std[names.index(k)] > 1e-8 for k in required)
    setcheck(
        "C1",
        "PARTIAL" if varied else "FAIL",
        "Initial arm/torso/head std measured; >1e-8 avoids float summation artifacts. Physical pose realism requires renders",
    )
    planner_configs = [p.get("planner", {}) for p in profiles]
    has_noise = bool(planner_configs) and all(
        p.get("action_noise", 0) > 0 and p.get("noise_hold", 0) > 0
        for p in planner_configs
    )
    setcheck(
        "C3",
        "PARTIAL" if has_noise else "FAIL",
        (
            "Execution-noise metadata present; recovery not measured"
            if has_noise
            else "No nonzero action_noise/noise_hold in source profile; waypoint jitter does not qualify"
        ),
    )
    verdict(
        "D2",
        all(
            max(e["roll_span"]) <= np.pi + 1e-5
            and min(e["roll_min"] + e["roll_command_min"]) >= -np.pi - 1e-5
            and max(e["roll_max"] + e["roll_command_max"]) <= np.pi + 1e-5
            for e in per_episode
        ),
        "Measured native roll positions and absolute roll commands; no modulo wrapping",
    )
    combos = Counter(
        tuple(c) for e in per_episode for c in e["close_sign_combinations"]
    )
    result["close_sign_counts"] = {str(k): v for k, v in combos.items()}
    setcheck(
        "D3",
        "PARTIAL",
        "Close-command sign combinations measured; separate object grasps from drawer-handle grasps before applying 95% gate",
    )
    setcheck(
        "D4",
        (
            "PARTIAL"
            if all(e["max_continuous_turn_rad"] <= np.pi + 1e-5 for e in per_episode)
            else "FAIL"
        ),
        "Command-sign segments; measured actual yaw. Totals retained for comparison; only 10Hz sampled states",
    )
    reverse_fraction = sum(e["reverse_frames"] > 0 for e in per_episode) / n
    result["reverse_episode_fraction"] = reverse_fraction
    verdict(
        "D5",
        reverse_fraction <= 0.1
        and all(e["base_direction_changes"] <= 2 for e in per_episode),
        "Reverse fraction and changes computed literally; task-specific exceptions are not assumed",
    )
    d6_ok = all(e["D6_pause_check"]["passed"] for e in per_episode)
    exception_used = any(e["D6_pause_check"]["exception"]
                         and not e["D6_pause_check"]["raw"]["passed"] for e in per_episode)
    setcheck("D6", ("EXCEPTION" if exception_used else "PASS") if d6_ok else "FAIL",
             "Literal metrics retained; only the owner-approved initial 50 policy frames "
             "of SeasonDish cue observation may be excluded; all other pauses checked")
    raw_switches = [e for e in raw_checks if "odd_gripper_switches" in e]
    if len(raw_switches) == n:
        verdict(
            "D7",
            all(not e["odd_gripper_switches"] for e in raw_switches),
            "Switch parity measured on original 20Hz oracle, not on already held-action replay",
        )
    feat = info["features"]
    image_keys = {k for k in feat if k.startswith("observation.images.")}
    expected_keys = {f"observation.images.{c}" for c in CAMS}
    verdict(
        "E3",
        image_keys == expected_keys
        and all(feat[k]["shape"] == CAMS[k.split(".")[-1]] for k in image_keys)
        and bool(profiles)
        and all(
            p.get("cameras")
            == {
                c: dict(width=v[1], height=v[0], fov=2.0 if c == "fetch_hand" else 1.5)
                for c, v in CAMS.items()
            }
            for p in profiles
        ),
        "Dataset feature shapes and pinned native camera profiles",
    )
    rb = root / "readback.json"
    if rb.exists():
        errors = [
            c["mean_absolute_error_255"]
            for e in read(rb).get("results", [])
            for c in e.get("rgb_checks", [])
            if "mean_absolute_error_255" in c
        ]
        result["video_sample_mae"] = dict(
            count=len(errors),
            mean=float(np.mean(errors)) if errors else None,
            max=max(errors, default=None),
        )
        if errors:
            setcheck(
                "E4",
                "FAIL" if max(errors) > 2 else "PARTIAL",
                "Stored readback samples only; full decoded-video comparison pending",
            )
    quality_path = root / "video_quality.json"
    if quality_path.exists():
        quality = read(quality_path)
        expected_frames = len(table) * len(CAMS)
        if quality.get("total_camera_frames") == expected_frames:
            verdict(
                "E4",
                quality["status"] == "success"
                and all(r["mean_absolute_error_255"] <= 2 for r in quality["results"]),
                "Full decoded-frame report, mean per episode/camera; video_quality.json",
            )
            result["full_video_quality"] = quality
    setcheck(
        "E5",
        "UNKNOWN",
        "No documented human review of >=10 complete three-camera episodes plus remaining contact sheets",
    )
    instructions = [m.get("instruction", "") for m in mappings.values()]
    setcheck(
        "F1",
        "PARTIAL",
        (
            "Instructions available for semantic review"
            if instructions
            else "Instructions not available"
        ),
    )
    tokens = [m.get("instruction_tokens") for m in mappings.values()]
    setcheck(
        "F2",
        (
            "PARTIAL"
            if tokens and all(t is not None and t <= 100 for t in tokens)
            else "UNKNOWN"
        ),
        "Stored PaliGemma counts; tokenizer recomputation pending",
    )
    if tokenizer is not None:
        import hashlib
        import sentencepiece as spm

        processor = spm.SentencePieceProcessor(model_file=str(tokenizer))
        recomputed = [
            len(processor.encode(text.strip() + "\n", add_bos=True))
            for text in instructions
        ]
        expected_hash = metadata.get("tokenizer_sha256")
        actual_hash = hashlib.sha256(Path(tokenizer).read_bytes()).hexdigest()
        verdict(
            "F2",
            bool(recomputed)
            and max(recomputed) <= 100
            and recomputed == tokens
            and expected_hash == actual_hash,
            {"recomputed_tokens": recomputed, "tokenizer_sha256": actual_hash},
        )
    verdict(
        "F3",
        info["fps"] == 10
        and all_raw
        and all(e["timestamp_error"] < 1e-4 for e in per_episode)
        and all(
            e["held_action_matches_export"]
            and e["control_steps"] // 2 == per_episode[i]["frames"]
            and (e.get("original_steps", e["control_steps"]) + 1) // 2
            == per_episode[i]["frames"]
            for i, e in enumerate(raw_checks)
        ),
        "All timestamps/actions/lengths checked: physical two-step hold replay, padding final odd source step",
    )
    sizes_ok = all(
        info.get(k, 0) > 0 for k in ["data_files_size_in_mb", "video_files_size_in_mb"]
    )
    setcheck(
        "G1",
        "PARTIAL" if sizes_ok else "FAIL",
        "info size fields >0; runtime loading not requested",
    )
    if load_samples:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset(
                repo_id=metadata.get("repository_id", "local/audit"),
                root=root,
                video_backend="pyav",
            )
            for i in [0, len(ds) - 1]:
                sample = ds[i]
                for c, s in CAMS.items():
                    assert tuple(sample[f"observation.images.{c}"].shape) == (
                        3,
                        s[0],
                        s[1],
                    )
            setcheck(
                "G1",
                "PARTIAL" if sizes_ok else "FAIL",
                f"First/last sample loaded in lerobot {importlib.metadata.version('lerobot')}; actual training commit still must match",
            )
        except Exception as e:
            setcheck("G1", "FAIL", f"{type(e).__name__}: {e}")
    verdict(
        "G2",
        all(
            f"videos/observation.images.{c}/{suffix}" in cols
            for c in CAMS
            for suffix in [
                "chunk_index",
                "file_index",
                "from_timestamp",
                "to_timestamp",
            ]
        ),
        "All required video-offset columns present",
    )
    quantiles = ["q01", "q10", "q50", "q90", "q99"]
    verdict(
        "G3",
        all(
            k in stats.get(feature, {})
            for feature in ["action", "observation.state"]
            for k in quantiles + ["mean", "std", "min", "max", "count"]
        ),
        "Global quantile/statistic keys checked",
    )
    verdict(
        "G4",
        all(
            f"stats/{feature}/{q}" in cols
            for feature in ["action", "observation.state"]
            for q in quantiles
        ),
        "All required per-episode quantile columns checked",
    )
    required_meta = ["episode_seed", "success", "terminated", "truncated"]
    missing = [k for k in required_meta if k not in cols]
    rlds = root / "meta/source_rlds_metadata.json"
    if rlds.exists():
        md = read(rlds)
        missing += [
            k
            for k in [
                "episode_seeds",
                "episode_lengths",
                "episode_durations_s",
                "success_once",
                "reward_sums",
            ]
            if k not in md
        ]
    else:
        missing.append("meta/source_rlds_metadata.json")
    setcheck(
        "G5",
        "FAIL" if missing else "PARTIAL",
        {
            "missing": missing,
            "note": "Seed/value consistency must additionally be verified against original H5",
        },
    )
    if not missing:
        from .dataset_metadata import verify_dataset_metadata

        try:
            verify_dataset_metadata(root, metadata)
            setcheck(
                "G5",
                "PASS",
                "All per-episode values and source_rlds_metadata arrays verified against exact H5",
            )
        except (ValueError, KeyError, OSError) as error:
            setcheck("G5", "FAIL", str(error))
    if all_raw:
        verdict(
            "H1",
            all(e["final_success"] for e in raw_checks),
            "Recorded final environment success for each exported RGB H5 (not planner return flag)",
        )
        cut = any(e["truncated"] for e in raw_checks)
        result["horizon_fractions"] = [
            e["control_steps"] / e["horizon"] if e["horizon"] else None
            for e in raw_checks
        ]
        setcheck(
            "H5",
            "FAIL" if cut else "PARTIAL",
            "No truncated episodes; horizon fractions measured, 'noticeably below' needs explicit task margin",
        )
    outcomes = metadata.get("source_episode_outcomes", [])
    attempted = [e for e in outcomes if e.get("status") != "not_run"]
    successes = sum(e.get("success") is True for e in attempted)
    if attempted:
        result["collection_sr"] = {
            "successes": successes,
            "attempts": len(attempted),
            "failed_seeds": [
                e["scene_seed"] for e in attempted if e.get("success") is not True
            ],
        }
        verdict(
            "H2",
            successes / len(attempted) >= 0.6,
            "Source success / all attempted source seeds, before replay filtering",
        )
    test_label = "test" in root.name.lower()
    setcheck("H3", "PASS" if n >= 1000 else ("PARTIAL" if test_label else "FAIL"),
             f"{n}/1000 episodes; test label in directory name: {test_label}")
    setcheck(
        "H4",
        "UNKNOWN",
        "Locate and verify 100 successful validation seeds next to this dataset, with all collection attempts excluded",
    )
    if profiles:
        setcheck(
            "A3",
            (
                "PARTIAL"
                if all(
                    p["env_kwargs"]["control_mode"] == "pd_joint_pos" for p in profiles
                )
                else "FAIL"
            ),
            "Recorded source/export profile; eval launcher separately required",
        )
        setcheck(
            "A4",
            (
                "PARTIAL"
                if len({p["env_kwargs"]["sim_backend"] for p in profiles}) == 1
                else "FAIL"
            ),
            "Recorded backend agrees across exported episodes; evaluation configuration separately required",
        )
        setcheck(
            "A1",
            "PARTIAL",
            "Source profiles use ds_fetch; end-to-end robot file hashes require separate verification",
        )
        from .provenance import dataset_provenance_complete
        verdict("A2", dataset_provenance_complete(metadata),
                "Component commits bind the recorded snapshot, final converter and every input batch converter")
    result["counts"] = dict(Counter(v["status"] for v in checks.values()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-samples", action="store_true")
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    result = audit(args.dataset, args.load_samples, args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "counts": result.get("counts"),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
