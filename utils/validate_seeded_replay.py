"""Run planner episodes and validate them by action replay on fresh seeded envs.

Example::

    uv run python -m utils.validate_seeded_replay --num-episodes 10

The source run uses the existing planner contracts.  The validation pass creates
one fresh environment per recorded trajectory and replays actions only; it never
calls ``set_state_dict``.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import my_scenes  # noqa: F401  (registration side effects)
from mani_skill.utils.wrappers.record import RecordEpisode
from utils.episode_replay import EpisodeSpec, SeededEpisodeBuilder
from utils.logging_utils import PlannerLogger
from utils.mikasa.seeding import seed_everything
from utils.test_planner import (
    accepts_scene_idx,
    classify_result,
    load_planner,
    planner_module_name,
)


@dataclass(frozen=True)
class PlannerCase:
    env_id: str
    planner: str
    control_mode: str
    reconfigure: bool = False


PLANNER_CASES = (
    PlannerCase("MyRoboCasa-v1", "myrobocasa_planner", "pd_joint_pos", True),
    PlannerCase(
        "MyRoboCasa_TakeItBack-v1",
        "myrobocasa_takeitback_planner",
        "pd_joint_delta_pos",
        True,
    ),
    PlannerCase(
        "MyRoboCasa_TakeItBackTray-v1",
        "myrobocasa_takeitback_tray_planner",
        "pd_joint_delta_pos",
        True,
    ),
    PlannerCase(
        "MyRoboCasa_FridgeVeggies-v1",
        "myrobocasa_fridge_veggies_planner",
        "pd_joint_pos",
        True,
    ),
    PlannerCase("MikasaSeasonDish-v0", "season_dish_planner", "pd_joint_pos"),
    PlannerCase("MikasaWaterPlants-v0", "water_plants_planner", "pd_joint_pos"),
    PlannerCase(
        "MikasaCabinetRetrieval-v0", "cabinet_retrieval_planner", "pd_joint_pos"
    ),
    PlannerCase("MikasaCabinetSearch-v0", "cabinet_search_planner", "pd_joint_pos"),
    PlannerCase("MikasaDepthRecall-v1", "depth_recall_v1_planner", "pd_joint_pos"),
)


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, (list, tuple, np.ndarray)) and np.size(value) == 1:
        value = np.asarray(value).reshape(-1)[0]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    return bool(value)


def _env_kwargs(args, case: PlannerCase) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_envs": 1,
        "render_mode": "rgb_array",
        "obs_mode": args.obs_mode,
        "robot_uids": "ds_fetch",
        "control_mode": case.control_mode,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
    }
    if args.scene_idx is not None and accepts_scene_idx(case.env_id):
        kwargs["scene_idx"] = args.scene_idx
    return kwargs


def _reset_kwargs(case: PlannerCase, seed: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"seed": int(seed)}
    if case.reconfigure:
        kwargs["options"] = {"reconfigure": True}
    return kwargs


def _planner_call(planner_fn, env, seed: int):
    params = inspect.signature(planner_fn).parameters
    kwargs = {}
    for name, value in (
        ("debug", False),
        ("vis", False),
        ("info", False),
        ("blind", False),
    ):
        if name in params:
            kwargs[name] = value
    return planner_fn(env, seed, **kwargs)


def _trajectory_rows(run_dir: Path):
    json_path = run_dir / "trajectory.json"
    h5_path = run_dir / "trajectory.h5"
    if not json_path.exists() or not h5_path.exists():
        return [], h5_path
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    with h5py.File(h5_path, "r") as source:
        for episode in metadata.get("episodes", []):
            trajectory_id = f"traj_{episode['episode_id']}"
            if trajectory_id not in source:
                continue
            group = source[trajectory_id]
            actions = group["actions"]
            success = (
                bool(group["success"][-1])
                if "success" in group and len(group["success"])
                else False
            )
            truncated = bool(group["truncated"][-1]) if len(group["truncated"]) else False
            terminated = bool(group["terminated"][-1]) if len(group["terminated"]) else False
            rows.append(
                {
                    "episode": episode,
                    "trajectory_id": trajectory_id,
                    "steps": int(len(actions)),
                    "success": success,
                    "truncated": truncated,
                    "terminated": terminated,
                    "recorded_verdict": (
                        "success" if success else "truncated" if truncated else "missed"
                    ),
                }
            )
    return rows, h5_path


def _primary_row(rows, source_verdict: str):
    if not rows:
        return None
    if source_verdict == "success":
        successful = [row for row in rows if row["recorded_verdict"] == "success"]
        if successful:
            return successful[-1]
    return max(rows, key=lambda row: row["steps"])


def _patch_metadata(run_dir: Path, spec: EpisodeSpec, source_verdict: str, validation: dict):
    path = run_dir / "trajectory.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["episode_spec_schema_version"] = spec.schema_version
    data["episode_spec"] = spec.with_updates(source_verdict=source_verdict).to_dict()
    data["source_verdict"] = source_verdict
    data["action_replay"] = validation
    for episode in data.get("episodes", []):
        episode["episode_spec"] = spec.with_updates(
            source_verdict=source_verdict,
            trajectory_id=f"traj_{episode['episode_id']}",
        ).to_dict()
    path.write_text(
        json.dumps(_jsonable(data), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_one(case: PlannerCase, seed: int, episode_id: int, args) -> dict[str, Any]:
    run_dir = Path(args.output_dir) / case.planner / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = _env_kwargs(args, case)
    reset_kwargs = _reset_kwargs(case, seed)
    spec = EpisodeSpec(
        episode_id=episode_id,
        env_id=case.env_id,
        env_kwargs=copy.deepcopy(env_kwargs),
        reset_kwargs=copy.deepcopy(reset_kwargs),
        seed=seed,
        planner=case.planner,
        control_mode=case.control_mode,
    )

    base_env = None
    recorder = None
    logger = None
    source_verdict = "errored"
    error = None
    try:
        import gymnasium as gym

        base_env = gym.make(case.env_id, **env_kwargs)
        recorder = RecordEpisode(
            base_env,
            output_dir=str(run_dir),
            trajectory_name="trajectory",
            save_trajectory=True,
            save_video=False,
            save_on_reset=True,
            record_env_state=False,
            record_reward=False,
            source_type="motionplanning",
            source_desc="seeded planner run; replayed by actions on a fresh env",
        )
        logger = PlannerLogger(
            recorder,
            run_dir=run_dir,
            name="planner",
            log_freq=args.log_freq,
        )
        logger.record_episode_spec(spec)
        seed_everything(seed)
        planner_fn = load_planner(case.planner)
        result = _planner_call(planner_fn, logger, seed)
        source_verdict = classify_result(result)
        logger.log_event("verdict", success=source_verdict == "success", verdict=source_verdict)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        source_verdict = "errored"
        if logger is not None:
            logger.log_event("error", error)
    finally:
        if logger is not None:
            try:
                logger.close()
            except Exception as exc:
                error = error or f"close {type(exc).__name__}: {exc}"
        elif base_env is not None:
            base_env.close()

    rows, h5_path = _trajectory_rows(run_dir)
    validation: dict[str, Any] = {
        "validated": False,
        "matched": False,
        "replay_verdict": None,
        "replay_steps": 0,
        "trajectory_id": None,
        "skipped": None,
    }
    if source_verdict in {"no_plan", "errored"}:
        validation.update(
            matched=True,
            replay_verdict=source_verdict,
            skipped="planner did not return a replayable completed episode",
        )
    else:
        row = _primary_row(rows, source_verdict)
        if row is None:
            validation.update(skipped="no recorded actions")
        else:
            replay_spec = spec.with_updates(trajectory_id=row["trajectory_id"])
            replay = SeededEpisodeBuilder(replay_spec).replay(h5_path, row["trajectory_id"])
            validation.update(
                validated=True,
                matched=replay.verdict == source_verdict,
                replay_verdict=replay.verdict,
                replay_steps=replay.steps,
                trajectory_id=row["trajectory_id"],
            )
            if replay.error:
                validation["error"] = replay.error

    _patch_metadata(run_dir, spec, source_verdict, validation)
    return {
        "episode_id": episode_id,
        "seed": seed,
        "planner": case.planner,
        "env_id": case.env_id,
        "source_verdict": source_verdict,
        "replay": validation,
        "recorded_trajectories": len(rows),
        "error": error,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--planner", action="append", help="planner module; repeat for several")
    parser.add_argument(
        "--all-planners", action="store_true", help="run all registered planner cases"
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/seeded_replay"))
    parser.add_argument(
        "--obs-mode", default="none", help="observation mode; 'none' is fastest for action replay"
    )
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-backend", default="cpu")
    parser.add_argument("--scene-idx", type=int, default=0)
    parser.add_argument("--log-freq", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    wanted = set(args.planner or [])
    cases = [case for case in PLANNER_CASES if not wanted or case.planner in wanted]
    unknown = wanted - {case.planner for case in PLANNER_CASES}
    if unknown:
        raise SystemExit(f"unknown planner(s): {', '.join(sorted(unknown))}")
    if not cases:
        raise SystemExit("no planner cases selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        print(f"=== {case.planner} on {case.env_id} ===", flush=True)
        for i in range(args.num_episodes):
            seed = args.start_seed + i * args.seed_step
            result = run_one(case, seed, i, args)
            results.append(result)
            print(
                f"seed={seed} source={result['source_verdict']} "
                f"replay={result['replay']['replay_verdict']} "
                f"matched={result['replay']['matched']}",
                flush=True,
            )

    for case in cases:
        case_results = [row for row in results if row["planner"] == case.planner]
        (args.output_dir / case.planner / "replay_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "planner": case.planner,
                    "env_id": case.env_id,
                    "num_episodes": len(case_results),
                    "results": case_results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    summary = {
        "schema_version": 1,
        "num_episodes": args.num_episodes,
        "start_seed": args.start_seed,
        "seed_step": args.seed_step,
        "results": results,
    }
    (args.output_dir / "replay_summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for case in cases:
        rows = [row for row in results if row["planner"] == case.planner]
        source_hits = sum(row["source_verdict"] == "success" for row in rows)
        replay_hits = sum(row["replay"]["replay_verdict"] == "success" for row in rows)
        matched = sum(bool(row["replay"]["matched"]) for row in rows)
        validated = sum(bool(row["replay"]["validated"]) for row in rows)
        print(
            f"{case.planner}: source {source_hits}/{len(rows)}, "
            f"replay {replay_hits}/{len(rows)}, "
            f"validated {validated}/{len(rows)}, matched {matched}/{len(rows)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
