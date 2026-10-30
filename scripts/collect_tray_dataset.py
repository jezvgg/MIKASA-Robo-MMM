"""Collect successful tray trajectories and build one disk-bounded LeRobot dataset."""

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tray_episode_checkers import (  # noqa: E402
    CHECKER_VERSION,
    checker_rejection,
)
from scripts.collect_tray_demos import (  # noqa: E402
    ENV_ID,
    MOD,
    PLANNER,
    patch_trajectory_metadata,
    plan_seed,
)


def run(cmd, code, timeout, log):
    env = dict(__import__("os").environ, PYTHONPATH=str(code))
    try:
        result = subprocess.run(
            cmd, cwd=code, env=env, capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired as exc:
        output = f"TIMEOUT {exc}"
        result = None
    log.write_text(output)
    return output, 0 if result is None else result.returncode


def collect_raw(seed, args):
    raw = args.out / "raw" / f"seed{seed}"
    log = args.out / "logs" / f"plan_seed{seed}.log"
    if (raw / "trajectory.h5").exists() and (raw / "trajectory.json").exists():
        try:
            rejection = checker_rejection(args.out / "trace", seed)
        except (OSError, ValueError):
            rejection = "checker_failed"
        if rejection and rejection != "checker_failed":
            shutil.rmtree(raw, ignore_errors=True)
            return seed, rejection
        if rejection is None:
            return seed, "success"
        shutil.rmtree(raw, ignore_errors=True)  # no current-version proof; re-run seed
    if raw.exists():
        shutil.rmtree(raw)
    verdict = plan_seed(seed, args, args.out / "trace", log, raw)
    if verdict != "success":
        if raw.exists():
            shutil.rmtree(raw)
        return seed, verdict
    try:
        rejection = checker_rejection(args.out / "trace", seed)
    except (OSError, ValueError):
        rejection = "checker_failed"
    if rejection:
        shutil.rmtree(raw, ignore_errors=True)
        return seed, rejection
    try:
        for ext in ("h5", "json"):
            matches = [p for p in raw.glob(f"*.{ext}") if p.stem != "trajectory"]
            if len(matches) != 1:
                raise RuntimeError(f"expected one *.{ext} in {raw}, found {matches}")
            matches[0].rename(raw / f"trajectory.{ext}")
        patch_trajectory_metadata(raw, seed, args.out / "trace")
    except Exception:
        shutil.rmtree(raw, ignore_errors=True)
        return seed, "capture_failed"
    return seed, "success"


def replay_one(seed, args, batch_dir):
    raw = args.out / "raw" / f"seed{seed}"
    replay = batch_dir / "replays" / f"seed{seed}"
    replay.mkdir(parents=True, exist_ok=True)
    output, rc = run(
        MOD
        + [
            "utils.replay_rgb",
            str(raw),
            "--output-dir",
            str(replay),
            "--robot",
            "ds_fetch",
            "--stride",
            "2",
        ],
        args.code,
        args.timeout,
        batch_dir / "logs" / f"replay_seed{seed}.log",
    )
    if rc or not (replay / "trajectory.h5").exists():
        return seed, "replay_failed", output[-1000:]
    events = sorted((args.out / "trace").glob(f"seed_{seed}/*_events.jsonl"))
    _, rc = run(
        MOD
        + [
            "utils.target_visibility",
            str(raw),
            "--out",
            str(args.out / "vis" / f"seed{seed}.json"),
            "--stride",
            "2",
        ]
        + (["--events", str(events[0])] if events else []),
        args.code,
        args.timeout,
        batch_dir / "logs" / f"visibility_seed{seed}.log",
    )
    return seed, "success" if rc == 0 else "visibility_failed", ""


def convert_batch(seeds, batch_index, args):
    batch = args.out / "postpass" / f"batch_{batch_index:03d}"
    shutil.rmtree(batch / "replays", ignore_errors=True)
    shutil.rmtree(batch / "merged", ignore_errors=True)
    (batch / "logs").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.replay_workers) as pool:
        results = list(pool.map(lambda seed: replay_one(seed, args, batch), seeds))
    failures = [result for result in results if result[1] != "success"]
    if failures:
        raise RuntimeError(f"batch {batch_index} replay failures: {failures[:5]}")

    merged = batch / "merged"
    output, rc = run(
        MOD
        + ["utils.merge_replays", "--src", str(batch / "replays"), "--out", str(merged)],
        args.code,
        args.timeout,
        batch / "merge.log",
    )
    if rc or not (merged / "trajectory.h5").exists():
        raise RuntimeError(f"batch {batch_index} merge failed: {output[-2000:]}")

    dataset = batch / "lerobot"
    output, rc = run(
        MOD
        + [
            "utils.convert_to_lerobot_stream",
            "--traj-path",
            str(merged / "trajectory.h5"),
            "--output-dir",
            str(dataset),
            "--fps",
            "10",
            "--task-name",
            args.task_name,
        ],
        args.code,
        args.timeout,
        batch / "convert.log",
    )
    if rc or not (dataset / "meta" / "info.json").exists():
        raise RuntimeError(f"batch {batch_index} conversion failed: {output[-2000:]}")

    shutil.rmtree(batch / "replays")
    shutil.rmtree(merged)
    if not args.keep_raw:
        for seed in seeds:
            shutil.rmtree(args.out / "raw" / f"seed{seed}", ignore_errors=True)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=5000)
    parser.add_argument("--need", type=int, default=1200)
    parser.add_argument("--max-attempts", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--replay-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument(
        "--task-name",
        default="pick up the cup from the counter and put it on the tray",
    )
    args = parser.parse_args()
    args.code = args.code.resolve()
    args.out = args.out.resolve()
    for name in ("logs", "trace", "raw", "vis", "postpass"):
        (args.out / name).mkdir(parents=True, exist_ok=True)

    summary_path = args.out / "collection_summary.json"
    existing_summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    can_resume = bool(
        existing_summary
        and existing_summary.get("checker_version") == CHECKER_VERSION
        and existing_summary.get("successful_trajectories", 0) >= args.need
    )
    if can_resume:
        successes = existing_summary["success_seeds"][: args.need]
        for seed in successes:
            try:
                if checker_rejection(args.out / "trace", seed) is not None:
                    can_resume = False
                    break
            except (OSError, ValueError):
                can_resume = False
                break
    if can_resume:
        collection_summary = existing_summary
        print("RESUME_COLLECTION", len(successes), flush=True)
    else:
        shutil.rmtree(args.out / "postpass", ignore_errors=True)
        shutil.rmtree(args.out / "lerobot", ignore_errors=True)
        (args.out / "postpass").mkdir(parents=True, exist_ok=True)
        verdicts = {}
        successes = []
        next_seed = args.start_seed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while len(successes) < args.need and len(verdicts) < args.max_attempts:
                seeds = [
                    next_seed + i
                    for i in range(min(args.workers, args.max_attempts - len(verdicts)))
                ]
                next_seed += len(seeds)
                for seed, verdict in pool.map(lambda s: collect_raw(s, args), seeds):
                    verdicts[seed] = verdict
                    if verdict == "success":
                        successes.append(seed)
                    print("SEED", seed, verdict, flush=True)
                    if len(successes) >= args.need:
                        break

        successes = sorted(successes)[: args.need]
        if len(successes) < args.need:
            raise RuntimeError(
                f"only collected {len(successes)}/{args.need} successful trajectories"
            )
        collection_summary = {
            "checker_version": CHECKER_VERSION,
            "env_id": ENV_ID,
            "planner": PLANNER,
            "robot": "ds_fetch",
            "control_mode": "pd_joint_pos",
            "source_hz": 20,
            "dataset_hz": 10,
            "target_successes": args.need,
            "attempts": len(verdicts),
            "successful_trajectories": len(successes),
            "start_seed": args.start_seed,
            "success_seeds": successes,
            "verdict_counts": {
                verdict: sum(value == verdict for value in verdicts.values())
                for verdict in sorted(set(verdicts.values()))
            },
        }
        summary_path.write_text(json.dumps(collection_summary, indent=2) + "\n")

    batch_datasets = []
    for batch_index, start in enumerate(range(0, len(successes), args.batch_size)):
        batch_seeds = successes[start : start + args.batch_size]
        dataset = args.out / "postpass" / f"batch_{batch_index:03d}" / "lerobot"
        if (dataset / "meta" / "info.json").exists():
            print("BATCH_RESUME", batch_index, flush=True)
            batch_datasets.append(dataset)
            continue
        print("BATCH", batch_index, "seeds", batch_seeds[0], batch_seeds[-1], flush=True)
        batch_datasets.append(convert_batch(batch_seeds, batch_index, args))

    final_dataset = args.out / "lerobot"
    output, rc = run(
        MOD
        + [
            "utils.combine_lerobot_stream",
            "--output-dir",
            str(final_dataset),
            *map(str, batch_datasets),
        ],
        args.code,
        args.timeout,
        args.out / "combine.log",
    )
    if rc or not (final_dataset / "meta" / "info.json").exists():
        raise RuntimeError(f"final dataset combine failed: {output[-3000:]}")
    for dataset in batch_datasets:
        shutil.rmtree(dataset.parent, ignore_errors=True)

    visibility = {}
    for seed in successes:
        path = args.out / "vis" / f"seed{seed}.json"
        if path.exists():
            visibility[str(seed)] = json.loads(path.read_text())["summary"]
    target_visibility = [
        item["target_seen_any_cam_frac"]
        for item in visibility.values()
        if "target_seen_any_cam_frac" in item
    ]
    manifest = {
        **collection_summary,
        "dataset_dir": str(final_dataset),
        "batch_size": args.batch_size,
        "batch_count": len(batch_datasets),
        "task_name": args.task_name,
        "visibility_count": len(visibility),
        "target_visibility_min": min(target_visibility) if target_visibility else None,
        "target_visibility_mean": (
            sum(target_visibility) / len(target_visibility) if target_visibility else None
        ),
        "visibility": visibility,
    }
    (args.out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("DATASET", json.dumps({k: v for k, v in manifest.items() if k != "visibility"}), flush=True)


if __name__ == "__main__":
    main()
