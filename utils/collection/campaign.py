"""Resume a fixed candidate pool with bounded independent simulator workers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

from .contract import read_json, write_json
from .pipeline import PHASES, assert_signature, run_worker, summarize
from .profile import runtime_signature


def campaign(root, *, start_seed, num_seeds, purpose, jobs, through, timeout,
             skip_native_rgb=False):
    if not 1 <= jobs <= 8 or num_seeds < 1 or start_seed < 0:
        raise ValueError("Use 1–8 workers, a positive candidate count and nonnegative seeds")
    seeds = list(range(start_seed, start_seed + num_seeds))
    if (root / "run.json").exists():
        run = read_json(root / "run.json")
        if run["seeds"] != seeds or run["purpose"] != purpose:
            raise ValueError("Cannot change the candidate pool or purpose on resume")
        if run.get("skip_native_rgb", False) != skip_native_rgb:
            raise ValueError("Cannot change native RGB retention on resume")
        assert_signature(run)
    else:
        signature = runtime_signature()
        root.mkdir(parents=True, exist_ok=False)
        run = dict(version=1, purpose=purpose, seeds=seeds, signature=signature,
                   skip_native_rgb=skip_native_rgb,
                   scene=signature["profile"]["env_id"], env_kwargs=signature["profile"]["env_kwargs"],
                   created_utc=datetime.now(timezone.utc).isoformat())
        write_json(root / "run.json", run)
    phases = PHASES[:PHASES.index(through) + 1]
    if skip_native_rgb:
        # Native RGB is a qualification artifact. Production keeps the original
        # 20 Hz state H5 and renders the physically verified held-action H5.
        phases = tuple(phase for phase in phases if phase != "native_rgb")

    stop = Event()

    def process(seed):
        for phase in phases:
            if stop.is_set():
                return None
            try:
                result = run_worker(root, phase, seed, timeout)
            except BaseException:
                # Stop queued work immediately if metadata/storage itself fails.
                stop.set()
                raise
            if result["status"] != "success":
                break
        return seed

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(process, seed): seed for seed in seeds}
        try:
            for future in as_completed(futures):
                if future.result() is None:
                    continue
                report = summarize(root)  # One writer; completion order never changes seed membership.
                print(f"completed={len(report['attempted_seeds'])}/{len(seeds)} "
                      f"counts={report['successful_counts']}", flush=True)
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    return summarize(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--num-seeds", type=int, required=True)
    parser.add_argument("--purpose", choices=("development", "train", "validation"), required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--through", choices=PHASES, default="validated")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--skip-native-rgb", action="store_true",
                        help="Keep native state H5, render only the verified held-action RGB H5")
    args = parser.parse_args()
    campaign(args.output.resolve(), start_seed=args.start_seed, num_seeds=args.num_seeds,
             purpose=args.purpose, jobs=args.jobs, through=args.through, timeout=args.timeout,
             skip_native_rgb=args.skip_native_rgb)


if __name__ == "__main__":
    main()
