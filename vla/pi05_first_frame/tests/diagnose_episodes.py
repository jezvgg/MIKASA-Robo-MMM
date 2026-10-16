"""Per-step diagnostics of a served policy on SameDrawer: when each stage latch fires and why.

Runs the same loop as eval_samedrawer.py (same env, first-frame wrapper, executor and
clipping) and records, at every 20 Hz control step, the drawer openings, the task latches,
qpos and whether the apple is grasped or on the plate, and for every policy request the
raw (unclipped) action chunk. Per episode it reports when each latch first fired, and for
a sequence violation which rule fired it, how far the cued drawer was open, how long after
closing it happened and how far the base was from its start pose.

    REPO_DIR=... PYTHONPATH=$REPO_DIR $SIM_VENV/bin/python diagnose_episodes.py \
        --server 127.0.0.1:8100 --out DIR [--seeds validation] [--shard 0/4] \
        [--max-episodes N] [--replan-steps 25] [--binarize-gripper]
    diagnose_episodes.py --summarize DIR...

--binarize-gripper  executes sign(gripper) instead of the model's continuous value
                    (the dataset's gripper channel is only ever -1 or +1)

Writes DIR/episodes.jsonl (eval record + diagnostics) and DIR/steps/<seed>.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("REPO_DIR", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vla" / "pi05_first_frame"))

import eval_samedrawer as ev  # noqa: E402

LATCHES = ("closed_done", "apple_was_grasped", "apple_placement_started", "apple_done",
           "sequence_violated", "wrong_drawer_touched", "apple_retention_violated")
CHANNELS = ("sh_pan", "sh_lift", "ua_roll", "elbow", "fa_roll", "wr_flex", "wr_roll", "gripper",
            "head_pan", "head_tilt", "torso", "base_fwd", "base_yaw")
CONTROL_HZ = 20


def make_step_log(env):
    import gymnasium as gym

    class StepLog(gym.Wrapper):
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            self.rows = []
            self._log(info)
            return obs, info

        def step(self, action):
            out = self.env.step(action)
            self._log(out[4])
            return out

        def _log(self, info):
            base = self.env.unwrapped
            on_plate = info.get("apple_on_plate") if isinstance(info, dict) else None
            self.rows.append((
                ev._np(base.drawer_open_amounts())[0].astype(np.float32),
                np.array([bool(ev._np(getattr(base, n))[0]) for n in LATCHES]),
                ev._np(base.agent.robot.get_qpos())[0].astype(np.float32),
                bool(ev._np(base.agent.is_grasping(base.apple))[0]),
                bool(ev._np(on_plate)[0]) if on_plate is not None else False,
            ))

    return StepLog(env)


class ActionLog:
    """Records each raw chunk with the control step it was requested at; optionally binarizes the gripper."""

    def __init__(self, policy, steps, binarize_gripper: bool):
        self.policy, self.steps, self.binarize = policy, steps, binarize_gripper
        self.chunks: list[tuple[int, np.ndarray]] = []

    def get_server_metadata(self):
        return self.policy.get_server_metadata()

    def infer(self, observation: dict) -> dict:
        response = self.policy.infer(observation)
        actions = np.array(response["actions"], dtype=np.float32, copy=True)
        self.chunks.append((len(self.steps.rows) - 1, actions.copy()))
        if self.binarize:
            actions[:, 7] = np.where(actions[:, 7] >= 0, 1.0, -1.0)
        return {**response, "actions": actions}


def first(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def diagnose(rows, chunks, target: int, low: np.ndarray, high: np.ndarray, execute: int) -> dict:
    amounts = np.stack([r[0] for r in rows])
    flags = np.stack([r[1] for r in rows])
    qpos = np.stack([r[2] for r in rows])
    grasp = np.array([r[3] for r in rows])
    on_plate = np.array([r[4] for r in rows])
    at = {name: first(flags[:, i]) for i, name in enumerate(LATCHES)}
    seconds = {name: (None if step is None else round(step / CONTROL_HZ, 2)) for name, step in at.items()}
    out = {"latch_s": seconds, "apple_first_grasped_s": None if first(grasp) is None else round(first(grasp) / CONTROL_HZ, 2)}

    sv = at["sequence_violated"]
    if sv is not None:
        closed = at["closed_done"]
        cause = "reopen_after_close" if closed is not None and closed < sv else "apple_before_close"
        window = amounts[sv:sv + 2 * CONTROL_HZ, target]
        out["sequence_violation"] = {
            "cause": cause,
            "s_after_close": None if closed is None else round((sv - closed) / CONTROL_HZ, 2),
            "target_open_m": round(float(amounts[sv, target]), 4),
            "target_max_next_2s_m": round(float(window.max()), 4),
            "target_max_after_m": round(float(amounts[sv:, target].max()), 4),
            "base_from_start_m": round(float(np.linalg.norm(qpos[sv, :2] - qpos[0, :2])), 3),
            "apple_grasped_before": bool(grasp[:sv + 1].any()),
            "apple_on_plate_before": bool(on_plate[:sv + 1].any()),
        }

    executed = np.concatenate([c[:execute] for _, c in chunks]) if chunks else np.zeros((0, 13), np.float32)
    below, above = executed < low, executed > high
    out["executed_actions"] = int(len(executed))
    out["outside_bounds_by_channel"] = {
        name: [int(below[:, i].sum()), int(above[:, i].sum())]
        for i, name in enumerate(CHANNELS) if below[:, i].any() or above[:, i].any()}
    out["outside_bounds_max_excess"] = {
        name: round(float(max((low[i] - executed[:, i]).max(), (executed[:, i] - high[i]).max())), 4)
        for i, name in enumerate(CHANNELS) if below[:, i].any() or above[:, i].any()}
    g = executed[:, 7] if len(executed) else np.zeros(0)
    out["gripper"] = {
        "ambiguous_frac_abs_lt_0.5": round(float(np.mean(np.abs(g) < 0.5)), 4) if len(g) else None,
        "closed_frac": round(float(np.mean(g < 0)), 4) if len(g) else None,
    }
    arrays = dict(amounts=amounts, flags=flags, qpos=qpos, grasp=grasp, on_plate=on_plate,
                  request_step=np.array([s for s, _ in chunks], dtype=np.int32),
                  raw_actions=np.stack([c for _, c in chunks]) if chunks else np.zeros((0, 1, 13), np.float32))
    return out, arrays


def summarize_dirs(dirs: list[Path]) -> dict:
    records = {}
    for d in dirs:
        for r in ev.read_jsonl(d / "episodes.jsonl"):
            records[r["seed"]] = r
    records = sorted(records.values(), key=lambda r: r["seed"])
    base = ev.summarize(records)
    svs = [r["diag"]["sequence_violation"] for r in records if "sequence_violation" in r["diag"]]
    causes = {c: sum(s["cause"] == c for s in svs) for c in sorted({s["cause"] for s in svs})}
    reopen = [s for s in svs if s["cause"] == "reopen_after_close"]

    def q(values):
        values = [v for v in values if v is not None]
        return [round(float(np.percentile(values, p)), 3) for p in (10, 50, 90)] if values else None

    clip = {}
    for r in records:
        for name, (lo, hi) in r["diag"]["outside_bounds_by_channel"].items():
            c = clip.setdefault(name, [0, 0])
            c[0] += lo
            c[1] += hi
    executed = sum(r["diag"]["executed_actions"] for r in records)
    return {
        **{k: base[k] for k in ("episodes", "success", "success_rate", "wilson95", "stages")},
        "sequence_violation": {
            "episodes": len(svs), "by_cause": causes,
            "reopen_s_after_close_p10_50_90": q([s["s_after_close"] for s in reopen]),
            "reopen_target_open_m_p10_50_90": q([s["target_open_m"] for s in reopen]),
            "reopen_target_max_after_m_p10_50_90": q([s["target_max_after_m"] for s in reopen]),
            "reopen_base_from_start_m_p10_50_90": q([s["base_from_start_m"] for s in reopen]),
            "reopen_apple_grasped_before": sum(s["apple_grasped_before"] for s in reopen),
        },
        "latch_s_p10_50_90": {name: q([r["diag"]["latch_s"][name] for r in records]) for name in LATCHES},
        "apple_first_grasped_s_p10_50_90": q([r["diag"]["apple_first_grasped_s"] for r in records]),
        "outside_bounds": {"executed_actions": executed,
                           "by_channel_below_above": clip},
        "gripper_ambiguous_frac_mean": round(float(np.mean(
            [r["diag"]["gripper"]["ambiguous_frac_abs_lt_0.5"] for r in records])), 4) if records else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="127.0.0.1:8100")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=Path(os.environ.get("SAMEDRAWER_DATASET_DIR", ".")))
    parser.add_argument("--seeds", default="validation")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--replan-steps", type=int, default=25)
    parser.add_argument("--max-policy-steps", type=int, default=ev.DEFAULT_HORIZON_STEPS)
    parser.add_argument("--binarize-gripper", action="store_true")
    parser.add_argument("--summarize", type=Path, nargs="+")
    args = parser.parse_args()

    if args.summarize:
        print(json.dumps(summarize_dirs(args.summarize), indent=2))
        return
    if args.out is None:
        parser.error("--out is required")

    ev.check_torch_cuda()
    from openpi_client import websocket_client_policy
    from utils.collection.client import run_policy_episode
    from utils.collection.profile import make_env, runtime_signature
    from utils.mikasa.seeding import seed_everything

    profile = ev.load_profile_with_torch_build()
    signature = runtime_signature(profile)
    if (signature["code_sha256"], signature["engine_sha256"]) != (ev.EXPECTED_CODE_SHA256, ev.EXPECTED_ENGINE_SHA256):
        raise SystemExit("Simulator differs from the one that recorded the dataset")

    seeds, _ = ev.resolve_seeds(args.seeds, args.dataset_dir)
    shard, shards = (int(x) for x in args.shard.split("/"))
    seeds = seeds[shard::shards][: args.max_episodes]

    host, port = args.server.rsplit(":", 1)
    for name in ("no_proxy", "NO_PROXY"):
        os.environ[name] = ",".join(filter(None, [os.environ.get(name), "localhost", "127.0.0.1", "::1"]))
    steps_env = make_step_log(make_env({"signature": signature, "purpose": "validation"}, rgb=True))
    env = ev.make_recording_env(steps_env)
    actions_log = ActionLog(ev.connect_policy_server(websocket_client_policy, host, int(port)), steps_env,
                            args.binarize_gripper)
    policy = ev.FirstFramePolicy(actions_log, "reset")
    instruction = env.unwrapped.get_language_instruction()[0]
    low, high = (np.asarray(b, dtype=np.float32).reshape(-1)[:13] for b in (env.action_space.low, env.action_space.high))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "steps").mkdir(exist_ok=True)
    (args.out / "run.json").write_text(json.dumps({
        "argv": sys.argv, "seeds": seeds, "server_metadata": policy.get_server_metadata(),
        "action_low": low.tolist(), "action_high": high.tolist()}, indent=2, default=str))
    log = args.out / "episodes.jsonl"
    done = {r["seed"] for r in ev.read_jsonl(log)}
    try:
        for seed in seeds:
            if seed in done:
                continue
            start = time.time()
            seed_everything(seed)
            obs, _ = env.reset(seed=seed)
            policy.reset()
            actions_log.chunks = []
            result = run_policy_episode(env, policy, obs, instruction, max_policy_steps=args.max_policy_steps,
                                        execute_horizon=args.replan_steps, stop_on_success=True, clip_actions=True)
            record = ev.episode_record(env, seed, result, time.time() - start)
            target = record["target_drawer"]
            record["diag"], arrays = diagnose(steps_env.rows, actions_log.chunks, target, low, high, args.replan_steps)
            record["binarize_gripper"] = args.binarize_gripper
            np.savez_compressed(args.out / "steps" / f"{seed}.npz", **arrays)
            with log.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps({k: record[k] for k in ("seed", "target_drawer", "success", "status")}
                             | {"sv": record["diag"].get("sequence_violation")}), flush=True)
    finally:
        env.close()
    print(json.dumps(summarize_dirs([args.out]), indent=2))


if __name__ == "__main__":
    main()
