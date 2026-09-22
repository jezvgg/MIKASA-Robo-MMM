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
                    case["compartment"] = int(np.asarray(traj["env_states/cube_cab"][0]).item())
                    case["steps"] = len(traj["actions"])
                    case["duration_seconds"] = case["steps"]/20
                    case["reward_sum"] = float(traj["rewards"][:].sum())
                    case["head_target_span"] = np.ptp(traj["actions"][:,8:10], axis=0).tolist()
                    case["initial_base"] = traj["env_states/articulations/ds_fetch"][0,13:16].tolist()
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
    for key in ("compartment", "planned_search_length"):
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
    parser.add_argument("kind", choices=("collection", "assets"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.kind](args.root.resolve(), args.output.resolve())
