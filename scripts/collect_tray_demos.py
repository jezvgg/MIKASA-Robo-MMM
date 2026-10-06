"""Collect tray demos with a given MIKASA-Robo-MMM tree and convert them to LeRobot, inside one job.

Per seed, in its own process (fresh env, as the author's pipeline does):
  utils.test_planner -n 1 (state, cpu)  ->  utils.replay_rgb --stride 2 (20 Hz -> 10 Hz, RGB)
then utils.merge_replays and utils.convert_to_lerobot_stream --fps 10 over the successful seeds.
Seeds are tried in the given order, then onward from max+1, until `--need` successes.

Writes <out>/collect_summary.json with every seed's verdict and, per kept seed, the target
visibility summary of utils.target_visibility (<out>/vis/seed<N>.json has it per frame).

Usage (from any directory; --code is this repository's root):
  python scripts/collect_tray_demos.py --code . --out runs/tray --seeds 0 1 2 3 4 5 6 7 8 9 --need 10 --workers 8

The pipeline patches each episode's `episode_seed` and `source_seed` from the actual planner seed
before RGB replay and conversion; `validation_seeds.json` contains 100 held-out planner-solvable seeds.
"""

import argparse
import json
import os
import re
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

ENV_ID = "MyRoboCasa_TakeItBackTray-v1"
PLANNER = "myrobocasa_takeitback_tray_planner"


# The cluster's ManiSkill fork imports my_scenes from inside mani_skill, so a module that imports
# my_scenes first (utils.replay_rgb) hits a circular import; importing mani_skill first avoids it.
MOD = [sys.executable, "-c", "import sys, runpy, mani_skill; sys.argv = sys.argv[1:]; "
       "runpy.run_module(sys.argv[0], run_name='__main__', alter_sys=True)"]


def run(cmd, code, timeout, log):
    env = dict(os.environ, PYTHONPATH=str(code))
    try:
        p = subprocess.run(cmd, cwd=code, env=env, capture_output=True, text=True, timeout=timeout)
        out = p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        out = f"TIMEOUT {e}"
    log.write_text(out)
    return out


DEFAULT_ACTION_NOISE = 0.001
DEFAULT_NOISE_HOLD = 10


def plan_seed(seed, a, log_dir, log_path, traj_dir=None):
    cmd = [
        sys.executable, "-m", "utils.test_planner", "--scene", ENV_ID,
        "--planner", PLANNER, "-n", "1", "--start-seed", str(seed),
        "--control-mode", "pd_joint_pos", "--obs-mode", "state",
        "--sim-backend", "cpu", "--render-backend", "cpu",
        "--log-dir", str(log_dir),
    ]
    if traj_dir is not None:
        cmd += ["--traj-dir", str(traj_dir)]
    out = run(cmd, a.code, a.timeout, log_path)
    m = re.search(rf"EP 1/1 seed={seed}\s+(SUCCESS|FAILED)", out)
    return "timeout" if out.startswith("TIMEOUT") else (m.group(1).lower() if m else "error")


def patch_trajectory_metadata(run_dir, seed, trace_dir):
    path = run_dir / "trajectory.json"
    data = json.loads(path.read_text())
    noise = {"action_noise": DEFAULT_ACTION_NOISE, "noise_hold": DEFAULT_NOISE_HOLD}
    events = sorted(trace_dir.glob(f"seed_{seed}/*_events.jsonl"))
    if events:
        for line in events[0].read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "execution_noise":
                noise = {
                    "action_noise": event.get("action_noise", DEFAULT_ACTION_NOISE),
                    "noise_hold": event.get("noise_hold", DEFAULT_NOISE_HOLD),
                }
                break
    for episode in data.get("episodes", []):
        episode.update(episode_seed=int(seed), source_seed=int(seed), **noise)
    data.update(source_seed=int(seed), **noise)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def collect(seed, a):
    raw, rgb = a.out / "raw" / f"seed{seed}", a.out / "rgb" / f"seed{seed}"
    shutil.rmtree(raw, ignore_errors=True)
    logs = a.out / "logs"
    verdict = plan_seed(seed, a, a.out / "trace", logs / f"plan_seed{seed}.log", raw)
    if verdict != "success":
        return seed, verdict
    try:
        rejection = checker_rejection(a.out / "trace", seed)
    except (OSError, ValueError):
        rejection = "checker_failed"
    if rejection:
        shutil.rmtree(raw, ignore_errors=True)
        return seed, rejection
    # test_planner names the files by timestamp; replay_rgb reads trajectory.{h5,json}
    for ext in ("h5", "json"):
        (src,) = [f for f in raw.glob(f"*.{ext}") if f.stem != "trajectory"]
        src.rename(raw / f"trajectory.{ext}")
    patch_trajectory_metadata(raw, seed, a.out / "trace")
    run(MOD + ["utils.replay_rgb", str(raw), "--output-dir", str(rgb), "--robot", "ds_fetch",
               "--stride", "2"], a.code, a.timeout, logs / f"replay_seed{seed}.log")
    if not (rgb / "trajectory.h5").exists():
        return seed, "replay_failed"
    # which camera sees the cup / tray on every kept frame (diagnostic, not a verdict)
    events = sorted((a.out / "trace").glob(f"seed_{seed}/*_events.jsonl"))
    run(MOD + ["utils.target_visibility", str(raw), "--out", str(a.out / "vis" / f"seed{seed}.json"),
               "--stride", "2"] + (["--events", str(events[0])] if events else []),
        a.code, a.timeout, logs / f"vis_seed{seed}.log")
    return seed, "success"


def validate_seed(seed, a):
    verdict = plan_seed(
        seed, a, a.out / "validation_trace",
        a.out / "logs" / f"validation_seed{seed}.log",
    )
    if verdict != "success":
        return seed, verdict
    try:
        rejection = checker_rejection(a.out / "validation_trace", seed)
    except (OSError, ValueError):
        rejection = "checker_failed"
    return seed, rejection or verdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--need", type=int, default=10)
    p.add_argument("--max-extra", type=int, default=20, help="seeds beyond the list to try if some fail")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--validation-need", type=int, default=100)
    p.add_argument("--validation-start", type=int, default=None)
    p.add_argument("--validation-max-extra", type=int, default=200)
    p.add_argument("--task-name", default="pick up the cup from the counter and put it on the tray")
    a = p.parse_args()
    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    order = list(a.seeds) + list(range(max(a.seeds) + 1, max(a.seeds) + 1 + a.max_extra))
    verdicts = {}
    with ThreadPoolExecutor(a.workers) as pool:
        i = 0
        while sum(v == "success" for v in verdicts.values()) < a.need and i < len(order):
            missing = a.need - sum(v == "success" for v in verdicts.values())
            batch = order[i:i + (missing if verdicts else len(a.seeds))]
            i += len(batch)
            for seed, v in pool.map(lambda s: collect(s, a), batch):
                verdicts[seed] = v
                print("SEED", seed, v, flush=True)
    ok = [s for s in order if verdicts.get(s) == "success"][: a.need]
    # merge_replays reads every seed* dir under --src: link only the kept ones
    keep = a.out / "rgb_keep"
    shutil.rmtree(keep, ignore_errors=True)
    keep.mkdir()
    for s in ok:
        link = keep / f"seed{s}"
        if not link.exists():
            link.symlink_to(a.out / "rgb" / f"seed{s}")
    env = dict(os.environ, PYTHONPATH=str(a.code))
    r1 = subprocess.run(MOD + ["utils.merge_replays", "--src", str(keep), "--out", str(a.out / "merged")],
                        cwd=a.code, env=env)
    r2 = subprocess.run(MOD + ["utils.convert_to_lerobot_stream", "--traj-path",
                         str(a.out / "merged" / "trajectory.h5"), "--output-dir", str(a.out / "lerobot"), "--fps", "10",
                         "--task-name", a.task_name], cwd=a.code, env=env)

    validation = {"validation_seeds": [], "attempted": {}, "success_rate": None}
    if a.validation_need > 0:
        collection_seeds = set(verdicts)
        start = a.validation_start
        if start is None:
            start = max(collection_seeds or set(a.seeds)) + 1
        candidates = [
            seed for seed in range(start, start + a.validation_need + a.validation_max_extra)
            if seed not in collection_seeds
        ]
        validation_verdicts = {}
        with ThreadPoolExecutor(a.workers) as pool:
            for seed, verdict in pool.map(lambda s: validate_seed(s, a), candidates):
                validation_verdicts[seed] = verdict
                print("VALIDATION", seed, verdict, flush=True)
                if sum(v == "success" for v in validation_verdicts.values()) >= a.validation_need:
                    break
        valid = [
            seed for seed in candidates
            if validation_verdicts.get(seed) == "success"
        ][: a.validation_need]
        validation = {
            "validation_seeds": valid,
            "attempted": {str(k): v for k, v in validation_verdicts.items()},
            "success_rate": (
                len(valid) / len(validation_verdicts) if validation_verdicts else 0.0
            ),
        }
        (a.out / "validation_seeds.json").write_text(
            json.dumps({
                "checker_version": CHECKER_VERSION,
                "env_id": ENV_ID,
                "planner": PLANNER,
                "task_name": a.task_name,
                "collection_seeds": sorted(collection_seeds),
                **validation,
            }, indent=2) + "\n"
        )
    vis = {}
    for s in ok:
        f = a.out / "vis" / f"seed{s}.json"
        if f.exists():
            vis[str(s)] = json.loads(f.read_text())["summary"]
    summary = {"checker_version": CHECKER_VERSION,
               "code": str(a.code), "seeds_kept": ok, "visibility": vis,
               "verdicts": {str(k): v for k, v in verdicts.items()},
               "merge_rc": r1.returncode, "convert_rc": r2.returncode,
               "task_name": a.task_name, "validation": validation}
    (a.out / "collect_summary.json").write_text(json.dumps(summary, indent=2))
    print("COLLECT_SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
