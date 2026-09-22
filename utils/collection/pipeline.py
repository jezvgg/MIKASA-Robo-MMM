"""CabinetSearch: state-only attempts, action replay, then success-only RGB."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .contract import (ACTION_NAMES, ACTION_REPEAT, CAMERAS, CONTROL_FPS,
                       POLICY_FPS, episode_summary, read_json, recording_info, write_json)
from .client import as_numpy, execute_actions, scalar_bool
from .profile import REPO, load_profile, make_env, runtime_signature

PHASES = ("oracle", "native", "validated", "native_rgb", "rgb")


def episode_path(root, phase, seed):
    return Path(root) / phase / str(seed) / "trajectory.h5"


def configure_events(env, directory):
    """The existing planner's phase events, with the simulator step clock."""
    def log_event(event, message, **extra):
        payload = dict(step=int(as_numpy(env.unwrapped.elapsed_steps).item()),
                       event=event, message=message, **extra)
        with (directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(payload) + "\n")
    env.log_event = log_event


def record_env(env, path: Path):
    from mani_skill.utils.wrappers.record import RecordEpisode

    return RecordEpisode(
        env,
        output_dir=str(path.parent),
        trajectory_name=path.stem,
        save_trajectory=True,
        save_video=False,
        save_on_reset=False,
        source_type="motionplanning",
        record_env_state=True,
    )


def collection_contract(base, *, run: dict, seed: int, stage: str, source=None, source_steps=None) -> dict:
    instruction = base.get_language_instruction()[0]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("The task must supply a nonempty get_language_instruction()")
    native = stage in {"oracle", "native"}
    return {
        "version": 3,
        "stage": stage,
        "explicit_time_and_proprio": True,
        "signature_sha256": run["signature"]["code_sha256"],
        "profile": run["signature"]["profile"],
        "task_config": run["signature"]["task_config"],
        "scene_seed": seed,
        "planner_seed": seed,
        "waypoint_noise_seed": seed + run["signature"]["profile"]["planner"]["noise_seed_offset"],
        "action_units": ["rad"] * 7 + ["normalized_gripper"] + ["rad", "rad", "m", "normalized_forward_velocity", "normalized_yaw_velocity"],
        "base_velocity_scale": as_numpy(base.agent.controller.controllers["base"].config.upper).tolist(),
        "global_state_use": "debug_only_never_policy_input",
        "policy_fps": CONTROL_FPS if native else POLICY_FPS,
        "action_repeat": 1 if native else ACTION_REPEAT,
        "instruction": instruction,
        "action_names": ACTION_NAMES,
        "state_joint_names": [j.name for j in base.agent.robot.get_active_joints()],
        "source": source,
        "source_steps": source_steps,
        "resampling": "none" if native else "first_target_two_step_hold",
    }


def add_contract(path: Path, contract: dict) -> None:
    meta = read_json(path.with_suffix(".json"))
    with h5py.File(path, "a") as h5:
        trajectory = h5["traj_0"]
        n = len(trajectory["actions"])
        # ManiSkill Articulation.get_state: root pose 7 + velocities 6,
        # then the 15 qpos and 15 qvel values. Checked against rendered obs.
        state = trajectory["env_states/articulations/ds_fetch"]
        if state.shape != (n+1, 43):
            raise ValueError("Articulation state layout differs from the pinned engine")
        qpos = state[:, 13:28]
        for name, value in {"timestamp": np.arange(n+1) / CONTROL_FPS,
                            "qpos": qpos, "proprio": qpos[:, 3:], "global_state": qpos[:, :3]}.items():
            trajectory.create_dataset(name, data=value)
        summary = episode_summary(trajectory)
        contract.update(summary)
    for episode in meta["episodes"]:
        episode.update(success=summary["success"], success_once=summary["success_once"])
    meta["mikasa_data"] = contract
    write_json(path.with_suffix(".json"), meta)


def preserve_diagnostic(path):
    """Keep a failed state-only attempt for diagnosis, outside the accepted pool."""
    if not path.exists():
        return None
    diagnostic = path.with_name("failed-trajectory.h5")
    path.rename(diagnostic)
    if path.with_suffix(".json").exists():
        path.with_suffix(".json").rename(diagnostic.with_suffix(".json"))
    return diagnostic.name


def task_metrics(info: dict) -> dict:
    """Keep small numerical task diagnostics for every attempt, off-policy.

    Stage flags and search length explain failures and support split audits.
    Observations and privileged metrics never enter the policy input.
    """
    result = {}
    for key, value in info.items():
        if isinstance(value, dict):
            continue
        array = as_numpy(value)
        if array.dtype.kind not in "biuf" or array.size > 64:
            continue
        result[key] = array.item() if array.size == 1 else array.tolist()
    return result


class NonReplayableMotion(RuntimeError):
    """An oracle changed the physical robot without issuing a recorded action."""


@contextlib.contextmanager
def action_only_oracle(env):
    """Permit initialization during reset; reject direct robot state writes later.

    Changing even a continuous joint by 2*pi affects raw absolute targets and
    contact caches. Such a change has no action in H5, so it cannot be replayed.
    The guard follows robots recreated by reset and restores instance methods
    when collection ends. Planner-side kinematic models remain unrestricted.
    """
    missing = object()
    reset_override = vars(env).get("reset", missing)
    original_reset = env.reset
    installed = []

    def restore():
        for robot, name, previous in reversed(installed):
            if previous is missing:
                delattr(robot, name)
            else:
                setattr(robot, name, previous)
        installed.clear()

    def protect():
        robot = env.unwrapped.agent.robot
        for name in ("set_qpos", "set_qvel", "set_pose"):
            if not callable(getattr(robot, name, None)):
                continue
            previous = vars(robot).get(name, missing)

            def reject(*args, _name=name, **kwargs):
                raise NonReplayableMotion(
                    f"robot.{_name} outside reset is not represented by recorded actions"
                )

            setattr(robot, name, reject)
            installed.append((robot, name, previous))

    def reset(*args, **kwargs):
        restore()
        result = original_reset(*args, **kwargs)
        protect()
        return result

    env.reset = reset
    try:
        protect()
        yield
    finally:
        restore()
        if reset_override is missing:
            delattr(env, "reset")
        else:
            env.reset = reset_override


def collect_one(root: Path, run: dict, seed: int) -> dict:
    from utils.mikasa.seeding import seed_everything
    from utils.test_planner import classify_result, load_planner

    seed_everything(seed)
    path = episode_path(root, "oracle", seed)
    env = record_env(make_env(run), path)
    base = env.unwrapped
    success = False
    verdict, reason = "error", None
    try:
        seed_everything(seed)
        configure_events(env, path.parent)
        planner_cfg = run["signature"]["profile"]["planner"]
        try:
            with action_only_oracle(env):
                verdict = classify_result(load_planner(planner_cfg["module"])(
                    env, seed, waypoint_noise_seed=seed + planner_cfg["noise_seed_offset"],
                    waypoint_noise_m=planner_cfg["waypoint_noise_m"]))
        except NonReplayableMotion as exc:
            traceback.print_exc()
            verdict, reason = "non_replayable", str(exc)
        except Exception as exc:
            traceback.print_exc()
            verdict, reason = "error", f"{type(exc).__name__}: {exc}"
        info = base.get_info()
        success = verdict == "success" and scalar_bool(info["success"])
        metrics = task_metrics(info)
        steps = int(as_numpy(base.elapsed_steps).item())
        env.flush_trajectory(save=steps > 0)
        contract = collection_contract(base, run=run, seed=seed, stage="oracle")
    finally:
        env.close()
    result = dict(status="success" if success else (verdict if verdict != "success" else "missed"),
                  success=success, steps=steps, duration_seconds=steps/CONTROL_FPS,
                  task_metrics=metrics)
    if reason:
        result["reason"] = reason
    if steps and path.exists():
        add_contract(path, contract)
        if success:
            recording_info(path, stage="oracle")
        else:
            result["diagnostic_h5"] = preserve_diagnostic(path)
    return result


def initial_state_matches(group: h5py.Group, state: dict) -> bool:
    """Validate reset equivalence; do not inject an oracle's later task answers."""
    if set(group) != set(state):
        return False
    for name, value in state.items():
        if isinstance(value, dict):
            if not initial_state_matches(group[name], value):
                return False
        else:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            expected = group[name][0]
            actual = np.asarray(value)[0]
            if expected.shape != actual.shape or not np.allclose(
                expected, actual, atol=1e-5, rtol=0
            ):
                return False
    return True


def validate_one(root: Path, run: dict, seed: int, *, native=False) -> dict:
    from utils.mikasa.seeding import seed_everything

    source = episode_path(root, "oracle", seed)
    recording_info(source, stage="oracle")
    stage = "native" if native else "validated"
    path = episode_path(root, stage, seed)
    seed_everything(seed)
    env = record_env(make_env(run), path)
    base = env.unwrapped
    success = False
    status = "native_replay_failed" if native else "resample_failed"
    try:
        env.reset(seed=seed)
        with h5py.File(source, "r") as h5:
            traj = h5["traj_0"]
            source_steps = len(traj["actions"])
            actions = traj["actions"][:] if native else traj["actions"][::ACTION_REPEAT]
            repeat = 1 if native else ACTION_REPEAT
            if not initial_state_matches(traj["env_states"], base.get_state_dict()):
                return {"status": "initial_state_mismatch"}
        result = execute_actions(env, actions, repeat=repeat)
        success = result["completed"] and result["success"]
        if not result["completed"]:
            status = result["status"]
        env.flush_trajectory(save=result["control_steps"] > 0)
        if result["control_steps"] > 0:
            contract = collection_contract(
                base, run=run, seed=seed,
                stage=stage,
                source=str(source.relative_to(root)),
                source_steps=source_steps,
            )
    finally:
        env.close()
    if result["control_steps"] > 0:
        add_contract(path, contract)
    if not success:
        diagnostic = preserve_diagnostic(path)
        return {
            "status": status,
            "success": False,
            "diagnostic_h5": diagnostic,
            "source_steps": source_steps,
            "steps": result["control_steps"],
            "task_metrics": task_metrics(result["info"]),
        }
    recording_info(path, stage=stage)
    return {
        "status": "success",
        "source_steps": source_steps,
        "steps": result["control_steps"],
        "padded_steps": result["control_steps"] - source_steps,
        "task_metrics": task_metrics(result["info"]),
    }


def read_state(group: h5py.Group, index: int) -> dict:
    # Restore the single-environment batch dimension removed by RecordEpisode.
    return {
        key: read_state(value, index)
        if isinstance(value, h5py.Group)
        else np.expand_dims(value[index], 0)
        for key, value in group.items()
    }


def render_one(
    root: Path, run: dict, seed: int, render_backend: str, *, native=False
) -> dict:
    """Render validated states. This is rendering, not another success check."""
    from utils.mikasa.seeding import seed_everything

    source_stage = "native" if native else "validated"
    stage = "native_rgb" if native else "rgb"
    source = episode_path(root, source_stage, seed)
    meta = recording_info(source, stage=source_stage)
    path = episode_path(root, stage, seed)
    seed_everything(seed)
    env = make_env(run, rgb=True, render_backend=render_backend)
    base = env.unwrapped
    try:
        env.reset(seed=seed)
        with h5py.File(source, "r") as src, h5py.File(path, "w") as dst:
            trajectory = src["traj_0"]
            src.copy(trajectory, dst, name="traj_0")
            output = dst["traj_0"]
            del output["obs"]
            n = len(trajectory["actions"]) + 1
            qpos = output.create_dataset("obs/agent/qpos", (n, 15), dtype="f4")
            images = {
                camera: output.create_dataset(
                    f"obs/sensor_data/{camera}/rgb",
                    (n, *shape),
                    dtype="u1",
                    chunks=(1, *shape),
                    compression="gzip",
                    compression_opts=1,
                )
                for camera, shape in CAMERAS.items()
            }
            for t in range(n):
                base.set_state_dict(read_state(trajectory["env_states"], t))
                base._elapsed_steps[:] = t
                obs = base.get_obs()
                if set(obs["sensor_data"]) != set(CAMERAS):
                    raise ValueError("Unexpected camera rig")
                qpos[t] = obs["agent"]["qpos"][0].cpu().numpy()
                if "qpos" in trajectory and not np.allclose(qpos[t], trajectory["qpos"][t], atol=1e-6, rtol=0):
                    raise ValueError("Restored qpos differs from source state at render time")
                for camera, shape in CAMERAS.items():
                    frame = obs["sensor_data"][camera]["rgb"][0].cpu().numpy()
                    if frame.shape != shape or frame.dtype != np.uint8:
                        raise ValueError(f"Unexpected image format: {camera}")
                    images[camera][t] = frame
    finally:
        env.close()
    meta["env_info"]["env_kwargs"].update(obs_mode="rgb", render_backend=render_backend)
    meta["mikasa_data"].update(
        stage=stage,
        source=str(source.relative_to(root)),
        rendering="saved_validated_states",
    )
    write_json(path.with_suffix(".json"), meta)
    recording_info(path, stage=stage)
    return {"status": "success", "steps": n - 1}


def assert_signature(run):
    if runtime_signature() != run["signature"]:
        raise ValueError("Source, robot, task settings or engine changed since run creation")


def run_worker(root, phase, seed, timeout=1800):
    directory = episode_path(root, phase, seed).parent
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    # A second observer must not restart a live simulation after its tool wait ends.
    with (directory / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Worker already running: {phase}/{seed}") from exc
        if result_path.exists():
            result = read_json(result_path)
            if result["status"] == "success":
                recording_info(directory / "trajectory.h5", stage=phase)
            return result
        if (directory / "worker.log").exists():
            # Preserve interrupted attempts instead of silently replacing evidence.
            archive = directory / ("interrupted-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
            archive.mkdir()
            for path in list(directory.iterdir()):
                if path.is_file() and path.name != "worker.lock":
                    path.rename(archive / path.name)
        command = [sys.executable, "-m", __package__ + ".pipeline", "_worker", str(root), phase, str(seed)]
        start = time.monotonic()
        with (directory / "worker.log").open("w") as log:
            try:
                process = subprocess.run(command, cwd=REPO, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=timeout)
                reason = f"worker_exit_{process.returncode}"
            except subprocess.TimeoutExpired:
                reason = "timeout"
        if not result_path.exists():
            write_json(result_path, dict(status="error", reason=reason, seed=seed,
                                        phase=phase, wall_seconds=time.monotonic() - start))
        result = read_json(result_path)
        print(f"{phase} seed={seed}: {result['status']} ({time.monotonic()-start:.1f}s)", flush=True)
        return result


def summarize(root):
    run = read_json(root / "run.json")
    outcomes = {phase: {} for phase in PHASES}
    for phase in PHASES:
        for seed in run["seeds"]:
            path = episode_path(root, phase, seed).parent / "result.json"
            if path.exists():
                outcomes[phase][str(seed)] = read_json(path)
    counts = {phase: sum(r["status"] == "success" for r in results.values())
              for phase, results in outcomes.items()}
    report = dict(version=1, purpose=run["purpose"], candidate_seeds=run["seeds"],
                  attempted_seeds=[int(s) for s in outcomes["oracle"]],
                  oracle_successful_seeds=[int(s) for s,r in outcomes["oracle"].items() if r["status"] == "success"],
                  successful_counts=counts, results=outcomes,
                  ready_seeds=[int(s) for s,r in outcomes["rgb"].items() if r["status"] == "success"])
    write_json(root / "attempts.json", report)
    return report


def worker(root, phase, seed):
    path = episode_path(root, phase, seed).parent / "result.json"
    start = time.monotonic()
    run = read_json(root / "run.json")
    result = {}
    try:
        assert_signature(run)
        if phase == "oracle":
            result = collect_one(root, run, seed)
        elif phase in {"native", "validated"}:
            result = validate_one(root, run, seed, native=phase == "native")
        else:
            result = render_one(root, run, seed,
                                run["signature"]["profile"]["env_kwargs"]["render_backend"],
                                native=phase == "native_rgb")
    except NonReplayableMotion as exc:
        traceback.print_exc()
        result = dict(status="non_replayable", reason=str(exc))
    except Exception as exc:
        traceback.print_exc()
        result = dict(status="error", reason=f"{type(exc).__name__}: {exc}")
    result.setdefault("success", result["status"] == "success")
    # A worker error may have no complete recording; do not invent its outcome.
    result.update(success_once=None, final_success=None, reward_sum=None)
    for name in ("trajectory.json", "failed-trajectory.json"):
        metadata_path = path.parent / name
        if metadata_path.exists():
            metadata = read_json(metadata_path).get("mikasa_data", {})
            if "success_once" in metadata:
                result.update(success_once=metadata["success_once"],
                              final_success=metadata["success"],
                              reward_sum=metadata["reward_sum"],
                              duration_seconds=metadata["duration_seconds"])
            break
    result.update(seed=seed, phase=phase, wall_seconds=time.monotonic() - start,
                  source_sha256=run["signature"]["code_sha256"],
                  finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    collect = subs.add_parser("collect")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--start-seed", type=int, required=True)
    collect.add_argument("--num-seeds", type=int, required=True)
    collect.add_argument("--purpose", choices=("development", "train", "validation"), required=True)
    collect.add_argument("--timeout", type=int, default=1800)
    prepare = subs.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--through", choices=PHASES, default="rgb")
    prepare.add_argument("--timeout", type=int, default=1800)
    sub = subs.add_parser("_worker")
    sub.add_argument("output", type=Path)
    sub.add_argument("phase", choices=PHASES)
    sub.add_argument("seed", type=int)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.command == "_worker":
        worker(root, args.phase, args.seed)
        return
    if args.command == "collect":
        if args.num_seeds < 1 or args.start_seed < 0:
            raise ValueError("Positive candidate count and nonnegative seeds required")
        signature = runtime_signature()
        root.mkdir(parents=True, exist_ok=False)
        run = dict(version=1, purpose=args.purpose,
                   seeds=list(range(args.start_seed, args.start_seed + args.num_seeds)),
                   signature=signature, scene=signature["profile"]["env_id"],
                   env_kwargs=signature["profile"]["env_kwargs"],
                   created_utc=datetime.now(timezone.utc).isoformat())
        write_json(root / "run.json", run)
        phases = ("oracle",)
    else:
        run = read_json(root / "run.json")
        assert_signature(run)
        phases = PHASES[:PHASES.index(args.through) + 1]
    for seed in run["seeds"]:
        for phase in phases:
            result = run_worker(root, phase, seed, args.timeout)
            summarize(root)
            if result["status"] != "success":
                break
    report = summarize(root)
    print(json.dumps(report["successful_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
