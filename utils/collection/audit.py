"""Read-only collection statistics and provenance of external kitchen assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .contract import read_json, write_json


def wilson(successes, attempts):
    if not attempts:
        return None
    z = 1.959963984540054
    p = successes / attempts
    den = 1 + z*z/attempts
    centre = (p + z*z/(2*attempts)) / den
    half = z * math.sqrt(p*(1-p)/attempts + z*z/(4*attempts*attempts)) / den
    return [centre-half, centre+half]


def stats(values):
    return None if not values else dict(count=len(values), mean=float(np.mean(values)),
        median=float(np.median(values)), minimum=float(min(values)), maximum=float(max(values)))


def collection(root, output):
    run = read_json(root / "run.json")
    phases = {}
    for phase in ("oracle", "native", "validated", "native_rgb", "rgb"):
        phases[phase] = {}
        for seed in run["seeds"]:
            path = root / phase / str(seed) / "result.json"
            if path.exists():
                phases[phase][seed] = read_json(path)
    source = phases["oracle"]
    complete = len(source) == len(run["seeds"]) and not (root / "INTERRUPTED.json").exists()
    n_ok = sum(r["status"] == "success" for r in source.values())
    report = dict(purpose=run["purpose"], candidate_count=len(run["seeds"]),
        completed_source_attempts=len(source), source_pool_complete=complete,
        source_sha256=run["signature"]["code_sha256"],
        phases={phase: dict(attempts=len(results), outcomes=dict(Counter(r["status"] for r in results.values())),
                            worker_seconds=stats([r["wall_seconds"] for r in results.values()]))
                for phase, results in phases.items()},
        expert_successes=n_ok, expert_success_rate=n_ok/len(source) if complete and source else None,
        expert_sr_wilson_95=wilson(n_ok, len(source)) if complete else None,
        native_losses=sum(r["status"] != "success" for r in phases["native"].values()),
        resampling_losses=sum(r["status"] != "success" for r in phases["validated"].values()),
        cases=[])
    for seed, result in sorted(source.items()):
        path = root / "oracle" / str(seed) / "trajectory.h5"
        if not path.exists():
            path = path.with_name("failed-trajectory.h5")
        case = dict(seed=seed, source_status=result["status"], metrics=result.get("task_metrics", {}))
        if path.exists():
            with h5py.File(path, "r") as h5:
                if "traj_0" in h5:
                    traj = h5["traj_0"]
                    if "cube_cab" in traj["env_states"]:
                        case["compartment"] = int(np.asarray(traj["env_states/cube_cab"][0]).item())
                    if "target_drawer" in traj["env_states"]:
                        case["target_drawer"] = int(np.asarray(traj["env_states/target_drawer"][0]).item())
                    if "target_is_shaker" in traj["env_states"]:
                        case["target_is_shaker"] = bool(np.asarray(traj["env_states/target_is_shaker"][0]).item())
                        case["station_left_is_shaker"] = bool(np.asarray(traj["env_states/station_left_is_shaker"][0]).item())
                    case["steps"] = len(traj["actions"])
                    case["duration_seconds"] = case["steps"]/20
                    case["reward_sum"] = float(traj["rewards"][:].sum())
                    case["head_target_span"] = np.ptp(traj["actions"][:,8:10], axis=0).tolist()
                    case["initial_base"] = traj["env_states/articulations/ds_fetch"][0,13:16].tolist()
                    case["initial_robot_root_pose"] = traj["env_states/articulations/ds_fetch"][0,:7].tolist()
            event_path = path.parent / "events.jsonl"
            events = [json.loads(line) for line in event_path.read_text().splitlines()] if event_path.exists() else []
            route = next((e.get("route") for e in events if e["message"] == "episode"), None)
            if isinstance(route, list) and "compartment" in case:
                case["planned_search_length"] = route.index(case["compartment"]) + 1
            case["last_phase"] = events[-1]["message"] if events else None
        case["phase_status"] = {phase: results[seed]["status"] if seed in results else "not_run"
                                for phase, results in phases.items()}
        report["cases"].append(case)
    report["coverage"] = {}
    for key in ("compartment", "planned_search_length", "target_is_shaker", "station_left_is_shaker", "target_drawer"):
        report["coverage"][key] = {
            "all_attempts": dict(Counter(str(c[key]) for c in report["cases"] if key in c)),
            "expert_successes": dict(Counter(str(c[key]) for c in report["cases"] if key in c and c["source_status"] == "success")),
            "ten_hz_successes": dict(Counter(str(c[key]) for c in report["cases"] if key in c and c["phase_status"]["validated"] == "success")),
        }
    report["successful_simulation_seconds"] = stats([c["duration_seconds"] for c in report["cases"]
        if c["source_status"] == "success" and "duration_seconds" in c])
    report["interpretation"] = ("Source SR counts all completed candidates before filtering. Incomplete/interrupted pools do not receive an SR estimate. Replay losses and coverage are reported separately; successful selection can change the task distribution.")
    write_json(output, report)
    print(json.dumps({key: report[key] for key in ("candidate_count", "completed_source_attempts", "expert_successes", "expert_success_rate", "native_losses", "resampling_losses")}))
    return report



def check(root, output):
    """Check completed H5, phase lineage and task waypoint-noise samples."""
    from .contract import (CAMERAS, check_actions, episode_summary, held_actions,
                           recording_info)

    run = read_json(root / "run.json")
    signature = run["signature"]
    records, noise_counts = [], Counter()
    unknown, source_results = [], 0
    for seed in run["seeds"]:
        source_actions = None
        oracle = root / "oracle" / str(seed) / "trajectory.h5"
        if oracle.exists() and (oracle.parent / "result.json").exists():
            if read_json(oracle.parent / "result.json")["status"] == "success":
                with h5py.File(oracle, "r") as h5:
                    source_actions = h5["traj_0/actions"][:]
        for phase in ("oracle", "native", "validated", "native_rgb", "rgb"):
            directory = root / phase / str(seed)
            result_path = directory / "result.json"
            if not result_path.exists():
                continue
            result = read_json(result_path)
            source_results += phase == "oracle"
            path = directory / ("trajectory.h5" if result["status"] == "success" else "failed-trajectory.h5")
            if not path.exists() or not path.with_suffix(".json").exists():
                if result["status"] == "success":
                    raise ValueError(f"Successful result has no recording: {directory}")
                unknown.append(dict(seed=seed, phase=phase, status=result["status"]))
                continue
            meta = read_json(path.with_suffix(".json"))
            contract = meta["mikasa_data"]
            if (contract["signature_sha256"] != signature["code_sha256"]
                    or contract["profile"] != signature["profile"]
                    or contract["scene_seed"] != seed
                    or meta["episodes"][0]["episode_seed"] != seed):
                raise ValueError(f"Mixed recording provenance: {path}")
            if result["status"] == "success":
                recording_info(path, stage=phase)
            with h5py.File(path, "r") as h5:
                traj = h5["traj_0"]
                actions = traj["actions"][:]
                check_actions(actions)
                n = len(actions)
                qpos = traj["qpos"][:]
                np.testing.assert_array_equal(qpos, traj["env_states/articulations/ds_fetch"][:,13:28])
                np.testing.assert_array_equal(traj["proprio"][:], qpos[:,3:])
                np.testing.assert_array_equal(traj["global_state"][:], qpos[:,:3])
                np.testing.assert_allclose(traj["timestamp"][:], np.arange(n+1)/20, atol=1e-9, rtol=0)
                if qpos.shape != (n+1,15) or not np.isfinite(qpos).all():
                    raise ValueError(f"Invalid proprioception: {path}")
                for key in ("success", "rewards", "terminated", "truncated"):
                    if traj[key].shape != (n,) or not np.isfinite(traj[key][:]).all():
                        raise ValueError(f"Invalid per-step {key}: {path}")
                summary = episode_summary(traj)
                if any(contract.get(k) != v for k,v in summary.items()):
                    raise ValueError(f"H5 summary mismatch: {path}")
                if bool(result.get("final_success")) != summary["success"]:
                    raise ValueError(f"Worker final success disagrees with H5: {path}")
                if result.get("success_once") != summary["success_once"]:
                    raise ValueError(f"Worker ever-success disagrees with H5: {path}")
                if phase != "oracle" and source_actions is not None and result["status"] == "success":
                    expected = source_actions if phase in {"native", "native_rgb"} else held_actions(source_actions)
                    np.testing.assert_array_equal(actions, expected)
                if phase in {"rgb", "native_rgb"}:
                    if set(traj["obs/sensor_data"]) != set(CAMERAS):
                        raise ValueError("Extra or missing RGB camera")
                    for camera, shape in CAMERAS.items():
                        sensor = traj[f"obs/sensor_data/{camera}"]
                        if set(sensor) != {"rgb"} or sensor["rgb"].shape != (n+1,*shape):
                            raise ValueError("Unexpected sensor fields or image dimensions")
                    np.testing.assert_allclose(traj["obs/agent/qpos"][:], qpos, atol=1e-6, rtol=0)
                cue = signature["task_config"].get("cue_steps", 0)
                after_cue = actions[cue:, 9] if signature["profile"]["env_id"] == "MikasaSeasonDish-v0" else []
                records.append(dict(seed=seed, phase=phase, status=result["status"], **summary,
                    head_target_span=np.ptp(actions[:,8:10],axis=0).tolist(),
                    post_cue_head_tilt_min=float(min(after_cue)) if len(after_cue) else None))
            if phase == "oracle" and signature["profile"]["env_id"] in {"MikasaSeasonDish-v0", "MikasaSameDrawer-v0", "MikasaDepthRecall-v1"}:
                events_path = directory / "events.jsonl"
                events = [json.loads(line) for line in events_path.read_text().splitlines()]
                samples = [event for event in events if event["message"] == "waypoint noise"]
                cfg = signature["profile"]["planner"]
                rng = np.random.default_rng(seed + cfg["noise_seed_offset"])
                for index, event in enumerate(samples):
                    expected = rng.uniform(-cfg["waypoint_noise_m"], cfg["waypoint_noise_m"], 3) * np.asarray(event["axes"], dtype=bool)
                    if event["sample_index"] != index:
                        raise ValueError("Nonconsecutive noise sample index")
                    np.testing.assert_array_equal(event["offset_m"], expected)
                    np.testing.assert_array_equal(event["goal_m"], np.asarray(event["original_m"])+expected)
                    noise_counts[event["waypoint"]] += 1
                if result["status"] == "success" and not samples:
                    raise ValueError("Successful oracle did not sample waypoint noise")
    report = dict(status="success", source_sha256=signature["code_sha256"],
        candidate_count=len(run["seeds"]), completed_source_attempts=source_results,
        source_pool_complete=source_results == len(run["seeds"]), checked_recordings=len(records),
        records=records, incomplete_or_missing_recordings=unknown,
        waypoint_samples_by_label=dict(noise_counts),
        noise_interpretation="Proposed goals, including rejected alternatives; not a count of executed motions.")
    write_json(output, report)
    print({key:report[key] for key in ("status", "source_pool_complete", "checked_recordings", "waypoint_samples_by_label")})
    return report


def assets(root, output):
    """Hash an external asset tree once; publish its manifest alongside a release."""
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        before = path.stat()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Asset changed while hashing: {path}")
        files[str(path.relative_to(root))] = dict(sha256=digest, bytes=after.st_size, mtime_ns=after.st_mtime_ns)
    content = {name: value["sha256"] for name, value in files.items()}
    result = dict(root=str(root), files=files, total_bytes=sum(v["bytes"] for v in files.values()),
                  content_sha256=hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest(),
                  captured_utc=datetime.now(timezone.utc).isoformat())
    write_json(output, result)
    print(f"Hashed {len(files)} asset files / {result['total_bytes']} bytes")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("collection", "assets", "check"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.kind](args.root.resolve(), args.output.resolve())
